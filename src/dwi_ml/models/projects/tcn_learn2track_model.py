# -*- coding: utf-8 -*-
"""
TCN model for tractography tracking.

Compatible with the Learn2Track trainer API:
- uses ModelWithDirectionGetter
- instantiates self.direction_getter
- returns (model_outputs, hidden_states, bundle_logits_per_line)
"""

import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_sequence
from torch.nn.utils.rnn import invert_permutation

from dwi_ml.data.processing.streamlines.post_processing import (
    compute_directions, normalize_directions, compute_n_previous_dirs
)
from dwi_ml.models.embeddings import NoEmbedding
from dwi_ml.models.main_models import (
    ModelWithPreviousDirections,
    ModelWithDirectionGetter,
    ModelWithNeighborhood,
    ModelWithOneInput
)
from dwi_ml.models.projects.learn2track_model import faster_unpack_sequence

logger = logging.getLogger("model_logger")


class TCNBlock(nn.Module):
    """Single TCN block with dilated causal convolution and residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )
        self.activation = nn.ReLU()

    def forward(self, x, mask=None):
        out = self.conv(x)

        # Causal trimming
        if self.conv.padding[0] > 0:
            out = out[:, :, :-self.conv.padding[0]]

        out = self.activation(out)
        out = self.dropout(out)

        res = x if self.residual is None else self.residual(x)
        if res.size(2) != out.size(2):
            res = res[:, :, -out.size(2):]

        out = out + res

        # Important: mask after residual addition.
        if mask is not None:
            out = out * mask[:, None, :].float()

        return out


class TCN(nn.Module):
    """Stack of TCN blocks."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 kernel_size=3, num_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = hidden_channels if i < num_layers - 1 else out_channels
            layers.append(
                TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
        self.network = nn.Sequential(*layers)
        self.output_size = out_channels

    def forward(self, x, mask=None):
        for block in self.network:
            x = block(x, mask)
        return x


class TCNLearn2TrackModel(
    ModelWithPreviousDirections,
    ModelWithDirectionGetter,
    ModelWithNeighborhood,
    ModelWithOneInput
):
    """
    TCN-based tracking model compatible with the Learn2Track infrastructure.
    """

    def __init__(self,
                 experiment_name,
                 step_size: Union[float, None],
                 compress_lines: Union[float, None],
                 nb_features: int,

                 # PREVIOUS DIRECTIONS
                 nb_previous_dirs: Union[int, None],
                 prev_dirs_embedded_size: Union[int, None],
                 prev_dirs_embedding_key: Union[str, None],
                 normalize_prev_dirs: bool,

                 # INPUT EMBEDDING
                 input_embedding_key: str,
                 input_embedded_size: Union[int, None],
                 nb_cnn_filters: Optional[List[int]],
                 kernel_size: Optional[Union[int, List[int]]],

                 # TCN
                 tcn_hidden_size: int,
                 tcn_num_layers: int,
                 tcn_kernel_size: int,

                 # DIRECTION GETTER
                 dg_key: str,
                 dg_args: Union[dict, None],

                 # NEIGHBORHOOD
                 neighborhood_type: Optional[str] = None,
                 neighborhood_radius: Optional[int] = None,
                 neighborhood_resolution: Optional[float] = None,
                 log_level=logging.root.level,
                 nb_points: Optional[int] = None,

                 # BUNDLE OPTIONS
                 use_bundle_ids: bool = False,
                 bundle_emb_dim: Optional[int] = None,
                 num_bundles: Optional[int] = None,
                 predict_bundle_ids: bool = False,

                 # REGULARIZATION
                 dropout: float = 0.1):

        
        super().__init__(
            experiment_name=experiment_name,
            step_size=step_size,
            nb_points=nb_points,
            compress_lines=compress_lines,
            log_level=log_level,

            # Neighborhood
            neighborhood_type=neighborhood_type,
            neighborhood_radius=neighborhood_radius,
            neighborhood_resolution=neighborhood_resolution,

            # Input embedding
            nb_features=nb_features,
            input_embedding_key=input_embedding_key,
            input_embedded_size=input_embedded_size,
            nb_cnn_filters=nb_cnn_filters,
            kernel_size=kernel_size,

            # Previous directions
            nb_previous_dirs=nb_previous_dirs,
            prev_dirs_embedded_size=prev_dirs_embedded_size,
            prev_dirs_embedding_key=prev_dirs_embedding_key,
            normalize_prev_dirs=normalize_prev_dirs,

            # For super ModelForTracking
            dg_args=dg_args,
            dg_key=dg_key
        )

        self.dropout = dropout
        self.log_level = log_level
        self.dg_args = dg_args

        self.use_bundle_ids = bool(use_bundle_ids)
        self.predict_bundle_ids = bool(predict_bundle_ids)

        if self.use_bundle_ids:
            if bundle_emb_dim is None:
                raise ValueError(
                    "bundle_emb_dim must be provided when use_bundle_ids=True"
                )
            if num_bundles is None:
                raise ValueError(
                    "num_bundles must be provided when use_bundle_ids=True"
                )

            self.bundle_emb_dim = int(bundle_emb_dim)
            self.num_bundles = int(num_bundles)

            if self.bundle_emb_dim <= 0:
                raise ValueError(
                    f"bundle_emb_dim must be > 0 (got {self.bundle_emb_dim})"
                )
            if self.num_bundles <= 0:
                raise ValueError(
                    f"num_bundles must be > 0 (got {self.num_bundles})"
                )

            self.bundle_emb = nn.Embedding(
                self.num_bundles, self.bundle_emb_dim
            )
        else:
            self.bundle_emb_dim = 0
            self.num_bundles = 0
            self.bundle_emb = None

        if dropout < 0 or dropout > 1:
            raise ValueError("The dropout rate must be between 0 and 1.")

        self.embedding_dropout = nn.Dropout(self.dropout)

        # Raw size before optional input embedding
        self.raw_input_size = nb_features * self.nb_neighbors

        # Size entering the TCN
        self.input_size = self.computed_input_embedded_size
        if self.use_bundle_ids:
            self.input_size += self.bundle_emb_dim
        if self.nb_previous_dirs > 0:
            self.input_size += self.prev_dirs_embedded_size

        # TCN backbone
        self.tcn = TCN(
            in_channels=self.input_size,
            hidden_channels=tcn_hidden_size,
            out_channels=tcn_hidden_size,
            kernel_size=tcn_kernel_size,
            num_layers=tcn_num_layers,
            dropout=dropout
        )
        self.tcn_num_layers = tcn_num_layers
        self.tcn_kernel_size = tcn_kernel_size
        self.context_len = 1 + (tcn_kernel_size - 1) * (
            2 ** tcn_num_layers - 1
        )

        # Optional SSL head kept for future use.
        self.ssl_head = nn.Sequential(
            nn.Linear(self.tcn.output_size, self.tcn.output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.tcn.output_size, 3)
        )

        # Direction getter
        self.instantiate_direction_getter(self.tcn.output_size)

        # Optional bundle classifier head
        if self.predict_bundle_ids:
            if self.num_bundles <= 0:
                raise ValueError(
                    "num_bundles must be > 0 when predict_bundle_ids=True"
                )
            self.bundle_classifier = nn.Linear(
                self.tcn.output_size, self.num_bundles
            )
        else:
            self.bundle_classifier = None

    def set_context(self, context):
        assert context in ['training', 'validation', 'tracking', 'visu',
                           'preparing_backward']
        self._context = context

    @property
    def params_for_checkpoint(self):
        params = super().params_for_checkpoint
        params.update({
            'nb_features': int(self.nb_features),
            'tcn_hidden_size': self.tcn.network[-1].conv.out_channels,
            'tcn_num_layers': len(self.tcn.network),
            'tcn_kernel_size': self.tcn.network[0].conv.kernel_size[0],
            'dropout': self.dropout,
            'use_bundle_ids': self.use_bundle_ids,
            'bundle_emb_dim': self.bundle_emb_dim,
            'num_bundles': self.num_bundles,
            'predict_bundle_ids': self.predict_bundle_ids,
        })
        return params

    @property
    def computed_params_for_display(self):
        p = super().computed_params_for_display
        p['tcn_output_size'] = self.tcn.output_size
        return p

    def forward(self,
                x: List[torch.Tensor],
                input_streamlines: List[torch.Tensor] = None,
                bundle_ids: torch.Tensor = None,
                hidden_recurrent_states: List = None,
                return_hidden: bool = False,
                point_idx: int = None):
        """
        Returns
        -------
        model_outputs
            Output ready for compute_loss() or get_tracking_directions().
        out_hidden_recurrent_states
            Always None for TCN, kept for API compatibility.
        bundle_logits_per_line
            None unless bundle prediction head is enabled.
        """
        del hidden_recurrent_states

        if self.context is None:
            raise ValueError("Please set context before usage.")

        assert x[0].shape[-1] == self.raw_input_size, \
            "Not the expected input size! Should be {} but got {}.".format(
                self.raw_input_size, x[0].shape[-1])

        unsorted_indices = None
        sorted_indices = None

        if self.context != 'tracking':
            sort_lengths = torch.as_tensor([len(s) for s in x])
            _, sorted_indices = torch.sort(sort_lengths, descending=True)
            unsorted_indices = invert_permutation(sorted_indices)
            x = [x[i] for i in sorted_indices]
            if input_streamlines is not None:
                input_streamlines = [input_streamlines[i]
                                     for i in sorted_indices]

        dev = next(self.parameters()).device

        # Bundle IDs
        if self.use_bundle_ids:
            if self.context == 'tracking' or bundle_ids is None:
                bundle_ids = torch.zeros(
                    len(x), device=dev, dtype=torch.long
                )
            else:
                bundle_ids = torch.as_tensor(
                    bundle_ids, device=dev, dtype=torch.long
                ).view(-1)

            if bundle_ids.numel() == 1 and len(x) > 1:
                bundle_ids = bundle_ids.expand(len(x))

            if self.context != 'tracking':
                bundle_ids = bundle_ids[sorted_indices.to(bundle_ids.device)]

            if bundle_ids.numel() != len(x):
                raise ValueError(
                    f"bundle_ids must have one id per streamline: got "
                    f"{bundle_ids.numel()} for {len(x)} streamlines "
                    f"(context={self.context})."
                )
        else:
            bundle_ids = None

        # Previous directions
        n_prev_dirs = None
        if self.nb_previous_dirs > 0:
            if input_streamlines is None:
                raise ValueError(
                    "input_streamlines must be provided when "
                    "nb_previous_dirs > 0"
                )

            dirs = compute_directions(input_streamlines)
            if self.normalize_prev_dirs:
                dirs = normalize_directions(dirs)

            n_prev_dirs = compute_n_previous_dirs(
                dirs, self.nb_previous_dirs, point_idx=point_idx
            )
            n_prev_dirs = pack_sequence(
                n_prev_dirs, enforce_sorted=False
            )
            n_prev_dirs = self.prev_dirs_embedding(n_prev_dirs.data)
            n_prev_dirs = self.embedding_dropout(n_prev_dirs)

        # Bundle embeddings
        if self.use_bundle_ids and bundle_ids is not None:
            b = self.bundle_emb(bundle_ids)
            b_list = [b[i].expand(x[i].shape[0], -1) for i in range(len(x))]
        else:
            b_list = None

        # Pack input
        x_packed = pack_sequence(x, enforce_sorted=False)
        batch_sizes = x_packed.batch_sizes
        x_data = x_packed.data

        # Input embedding
        if self.input_embedding_key == 'cnn_embedding':
            from dwi_ml.data.processing.space.neighborhood import \
                unflatten_neighborhood
            x_data = unflatten_neighborhood(
                x_data, self.neighborhood_vectors,
                self.neighborhood_type, self.neighborhood_radius,
                self.neighborhood_resolution
            )

        x_data = self.input_embedding_layer(x_data)
        x_data = self.embedding_dropout(x_data)

        if n_prev_dirs is not None:
            x_data = torch.cat((x_data, n_prev_dirs), dim=-1)

        if b_list is not None:
            b_packed = pack_sequence(b_list, enforce_sorted=False)
            x_data = torch.cat((x_data, b_packed.data), dim=-1)

        expected_size = self.input_size
        got_size = x_data.shape[-1]
        if got_size != expected_size:
            raise ValueError(
                f"Wrong feature size before TCN: expected "
                f"{expected_size}, got {got_size}."
            )

        # Rebuild per-sequence list
        seq_features = faster_unpack_sequence(
            PackedSequence(
                x_data,
                batch_sizes,
                x_packed.sorted_indices,
                x_packed.unsorted_indices
            )
        )

        # Pad for TCN
        padded = pad_sequence(seq_features, batch_first=True)   # (B, T, F)
        batch_size, max_len, _ = padded.shape
        padded = padded.permute(0, 2, 1).contiguous()           # (B, F, T)

        mask = torch.zeros(
            batch_size, max_len, dtype=torch.bool, device=dev
        )
        seq_lengths = []
        for i, seq in enumerate(seq_features):
            cur_len = len(seq)
            seq_lengths.append(cur_len)
            mask[i, :cur_len] = True

        # TCN
        tcn_out = self.tcn(padded, mask)                        # (B, H, T)
        tcn_out = tcn_out.permute(0, 2, 1).contiguous()        # (B, T, H)

        # Recreate the PackedSequence.data layout:
        # at time step t, only the first batch_sizes[t] sequences are active.
        flat_out = torch.cat(
            [tcn_out[:batch_sizes[t], t, :] for t in range(len(batch_sizes))],
            dim=0
        )

        assert flat_out.shape[-1] == self.direction_getter.input_size, \
            "Expecting input to direction getter of size {}. Got {}.".format(
                self.direction_getter.input_size, flat_out.shape[-1]
            )

        model_outputs = self.direction_getter(flat_out)

        bundle_logits_per_line = None
        if self.bundle_classifier is not None:
            bundle_logits_per_line = [
                self.bundle_classifier(tcn_out[i, :seq_lengths[i], :])
                for i in range(batch_size)
            ]
            if self.context != 'tracking' and unsorted_indices is not None:
                bundle_logits_per_line = [
                    bundle_logits_per_line[i] for i in unsorted_indices
                ]

        # Unpack for non-tracking mode, exactly like Learn2Track
        if self.context != 'tracking':
            if 'gaussian' in self.dg_key or 'fisher' in self.dg_key:
                x1, x2 = model_outputs

                x2 = PackedSequence(x2, batch_sizes)
                x2 = faster_unpack_sequence(x2)
                x2 = [x2[i] for i in unsorted_indices]

                x1 = PackedSequence(x1, batch_sizes)
                x1 = faster_unpack_sequence(x1)
                x1 = [x1[i] for i in unsorted_indices]

                model_outputs = (x1, x2)
            else:
                model_outputs = PackedSequence(model_outputs, batch_sizes)
                model_outputs = faster_unpack_sequence(model_outputs)
                model_outputs = [model_outputs[i] for i in unsorted_indices]

        out_hidden_recurrent_states = None if not return_hidden else None
        return model_outputs, out_hidden_recurrent_states, bundle_logits_per_line

    def take_lines_in_hidden_state(self, hidden_states, lines_to_keep):
        del hidden_states, lines_to_keep
        return None