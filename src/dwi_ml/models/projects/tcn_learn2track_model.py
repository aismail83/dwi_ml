# -*- coding: utf-8 -*-
"""
Two-block TCN model for tractography tracking.

Main idea:
- Block 1 processes the streamline sequence.
- Block 1 predicts bundle classes point by point.
- These point-wise bundle logits are reused as extra features for Block 2.
- Block 2 predicts features used by the direction getter.

Returns:
    (model_outputs, hidden_states, bundle_logits_per_line)
"""

import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_sequence
from torch.nn.utils.rnn import invert_permutation
from dwi_ml.data.processing.space.neighborhood import unflatten_neighborhood
from dwi_ml.data.processing.streamlines.post_processing import (
    compute_directions, normalize_directions, compute_n_previous_dirs
)
from dwi_ml.models.main_models import (
    ModelWithPreviousDirections,
    ModelWithDirectionGetter,
    ModelWithNeighborhood,
    ModelWithOneInput
)
from dwi_ml.models.projects.learn2track_model import faster_unpack_sequence

logger = logging.getLogger("model_logger")


class Chomp1d(nn.Module):
    """Removes the extra right-padding added to keep causal convolutions."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class CausalConvBlock(nn.Module):
    """
    One causal dilated Conv1d block:
    Conv1d -> Chomp -> ReLU -> Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        self.chomp = Chomp1d(padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.chomp(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class TCNSubBlock(nn.Module):
    """
    A TCN sub-block composed of 5 causal dilated convolutions:
    dilations = [1, 3, 6, 12, 24]
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        kernel_size: int = 6,
        dilations=(1, 3, 6, 12, 24),
        dropout: float = 0.0
    ):
        super().__init__()

        layers = []
        current_in = in_channels
        for d in dilations:
            layers.append(
                CausalConvBlock(
                    in_channels=current_in,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout
                )
            )
            current_in = hidden_channels

        self.network = nn.Sequential(*layers)
        self.output_size = hidden_channels

        self.residual_proj = (
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
            if in_channels != hidden_channels else nn.Identity()
        )

        self.out_relu = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: (B, C, T)
        mask: (B, T)
        """
        if mask is not None:
            x = x * mask[:, None, :].float()

        residual = self.residual_proj(x)
        out = self.network(x)
        out = out + residual
        out = self.out_relu(out)

        if mask is not None:
            out = out * mask[:, None, :].float()

        return out


class TCNLearn2TrackModel(
    ModelWithPreviousDirections,
    ModelWithDirectionGetter,
    ModelWithNeighborhood,
    ModelWithOneInput
):
    """
    Two-block TCN model compatible with Learn2Track-like API.

    Block 1:
        input sequence -> TCN -> bundle logits per point

    Block 2:
        input = [embedded_input ; bundle_logits_per_point]
        -> TCN -> direction getter
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
                 predict_bundle_ids: bool = True,

                 # REGULARIZATION
                 dropout: float = 0.1):

        super().__init__(
            experiment_name=experiment_name,
            step_size=step_size,
            nb_points=nb_points,
            compress_lines=compress_lines,
            log_level=log_level,

            neighborhood_type=neighborhood_type,
            neighborhood_radius=neighborhood_radius,
            neighborhood_resolution=neighborhood_resolution,

            nb_features=nb_features,
            input_embedding_key=input_embedding_key,
            input_embedded_size=input_embedded_size,
            nb_cnn_filters=nb_cnn_filters,
            kernel_size=kernel_size,

            nb_previous_dirs=nb_previous_dirs,
            prev_dirs_embedded_size=prev_dirs_embedded_size,
            prev_dirs_embedding_key=prev_dirs_embedding_key,
            normalize_prev_dirs=normalize_prev_dirs,

            dg_args=dg_args,
            dg_key=dg_key
        )

        if dropout < 0 or dropout > 1:
            raise ValueError("The dropout rate must be between 0 and 1.")

        self.dropout = dropout
        self.log_level = log_level
        self.dg_args = dg_args

        self.use_bundle_ids = bool(use_bundle_ids)
        self.predict_bundle_ids = bool(predict_bundle_ids)

        self.tcn_hidden_size = tcn_hidden_size
        self.tcn_num_layers = tcn_num_layers
        self.tcn_kernel_size = tcn_kernel_size

        if self.predict_bundle_ids and (num_bundles is None or num_bundles <= 0):
            raise ValueError(
                "num_bundles must be provided and > 0 when predict_bundle_ids=True"
            )

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
            self.num_bundles = int(num_bundles) if num_bundles is not None else 0
            self.bundle_emb = None

        self.embedding_dropout = nn.Dropout(self.dropout)

        # Raw input size before optional embedding
        self.raw_input_size = nb_features * self.nb_neighbors

        # Embedded input size used as point feature x^(1)
        self.input_size = self.computed_input_embedded_size
        if self.use_bundle_ids:
            self.input_size += self.bundle_emb_dim
        if self.nb_previous_dirs > 0:
            self.input_size += self.prev_dirs_embedded_size

        self.dilations = (1, 3, 6, 12, 24)

        # -------------------------
        # BLOCK 
        # -------------------------
        self.tcn = TCNSubBlock(
            in_channels=self.input_size,
            hidden_channels=tcn_hidden_size,
            kernel_size=tcn_kernel_size,
            dilations=self.dilations,
            dropout=dropout
        )


        self.context_len = 1 + (tcn_kernel_size - 1) * sum(self.dilations)

        
        # Direction getter uses Block  output
        self.instantiate_direction_getter(self.tcn.output_size)

    def set_context(self, context):
        assert context in ['training', 'validation', 'tracking', 'visu',
                           'preparing_backward']
        self._context = context

    @property
    def params_for_checkpoint(self):
        params = super().params_for_checkpoint
        params.update({
            'nb_features': int(self.nb_features),
            'tcn_hidden_size': self.tcn_hidden_size,
            'tcn_num_layers': self.tcn_num_layers,
            'tcn_kernel_size': self.tcn_kernel_size,
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

    def _flatten_time_major(self, seq_tensor, batch_sizes):
        """
        Recreate PackedSequence.data layout from padded tensor.

        Parameters
        ----------
        seq_tensor: torch.Tensor
            Shape (B, T, F)
        batch_sizes: torch.Tensor
            Packed sequence batch sizes.

        Returns
        -------
        flat_out: torch.Tensor
            Shape (sum(lengths), F)
        """
        return torch.cat(
            [seq_tensor[:batch_sizes[t], t, :] for t in range(len(batch_sizes))],
            dim=0
        )

    def forward(self,
            x: List[torch.Tensor],
            input_streamlines: List[torch.Tensor] = None,
            bundle_ids: torch.Tensor = None,
            hidden_recurrent_states: List = None,
            return_hidden: bool = False,
            point_idx: int = None):

        del hidden_recurrent_states
        dev = next(self.parameters()).device

        if self.context is None:
            raise ValueError("Please set context before usage.")

        assert x[0].shape[-1] == self.raw_input_size, \
            "Not the expected input size! Should be {} but got {}.".format(
                self.raw_input_size, x[0].shape[-1])

        unsorted_indices = None
        sorted_indices = None

        # Training / validation: sort by length
        if self.context != 'tracking':
            sort_lengths = torch.as_tensor([len(s) for s in x])
            _, sorted_indices = torch.sort(sort_lengths, descending=True)
            unsorted_indices = invert_permutation(sorted_indices)
            x = [x[i] for i in sorted_indices]
            if input_streamlines is not None:
                input_streamlines = [input_streamlines[i] for i in sorted_indices]

        # -----------------------------------
        # Previous directions: ALWAYS full sequence for TCN
        # -----------------------------------
        n_prev_dirs = None
        if self.nb_previous_dirs > 0:
            if input_streamlines is None:
                raise ValueError(
                    "input_streamlines must be provided when nb_previous_dirs > 0"
                )

            dirs = compute_directions(input_streamlines)
            if self.normalize_prev_dirs:
                dirs = normalize_directions(dirs)

            # Important: keep full temporal alignment for TCN
            n_prev_dirs = compute_n_previous_dirs(
                dirs, self.nb_previous_dirs, point_idx=None
            )
            n_prev_dirs = pack_sequence(n_prev_dirs, enforce_sorted=False)
            n_prev_dirs = self.prev_dirs_embedding(n_prev_dirs.data)
            n_prev_dirs = self.embedding_dropout(n_prev_dirs)

        # -----------------------------------
        # Pack input
        # -----------------------------------
        x_packed = pack_sequence(x, enforce_sorted=False)
        batch_sizes = x_packed.batch_sizes
        x_data = x_packed.data

        # -----------------------------------
        # Input embedding
        # -----------------------------------
        if self.input_embedding_key == 'cnn_embedding':
            x_data = unflatten_neighborhood(
                x_data, self.neighborhood_vectors,
                self.neighborhood_type, self.neighborhood_radius,
                self.neighborhood_resolution
            )

        x_data = self.input_embedding_layer(x_data)
        x_data = self.embedding_dropout(x_data)

        if n_prev_dirs is not None:
            assert x_data.shape[0] == n_prev_dirs.shape[0], \
                f"x_data: {x_data.shape}, n_prev_dirs: {n_prev_dirs.shape}"
            x_data = torch.cat((x_data, n_prev_dirs), dim=-1)

        expected_size = self.input_size
        got_size = x_data.shape[-1]
        if got_size != expected_size:
            raise ValueError(
                f"Wrong feature size before Block 1: expected "
                f"{expected_size}, got {got_size}."
            )

        # Rebuild per-sequence list after embedding
        seq_features = faster_unpack_sequence(
            PackedSequence(
                x_data,
                batch_sizes,
                x_packed.sorted_indices,
                x_packed.unsorted_indices
            )
        )

        # -----------------------------------
        # Pad sequences
        # -----------------------------------
        padded_x1 = pad_sequence(seq_features, batch_first=True)  # (B, T, F1)
        batch_size, max_len, _ = padded_x1.shape

        mask = torch.zeros(
            batch_size, max_len, dtype=torch.bool, device=dev
        )
        seq_lengths = []
        for i, seq in enumerate(seq_features):
            cur_len = len(seq)
            seq_lengths.append(cur_len)
            mask[i, :cur_len] = True

        # -----------------------------------
        # TCN block
        # -----------------------------------
        tcn_in = padded_x1.permute(0, 2, 1).contiguous()   # (B, F, T)
        tcn_out = self.tcn(tcn_in)                   # (B, H, T)
        tcn_out = tcn_out.permute(0, 2, 1).contiguous()   # (B, T, H)

        # -----------------------------------
        # Direction getter input
        # -----------------------------------
        if point_idx is None:
            dg_in = self._flatten_time_major(tcn_out, batch_sizes)
        else:
            lengths_t = torch.as_tensor(seq_lengths, device=dev, dtype=torch.long)

            if point_idx < 0:
                gather_idx = lengths_t + point_idx   # e.g. -1 => last valid point
            else:
                gather_idx = torch.full(
                    (batch_size,), point_idx, device=dev, dtype=torch.long
                )

            gather_idx = torch.clamp(gather_idx, min=0)
            gather_idx = torch.minimum(gather_idx, lengths_t - 1)

            dg_in = tcn_out[
                torch.arange(batch_size, device=dev),
                gather_idx,
                :
            ]   # (B, H)

        assert dg_in.shape[-1] == self.direction_getter.input_size, \
            "Expecting input to direction getter of size {}. Got {}.".format(
                self.direction_getter.input_size, dg_in.shape[-1]
            )

        model_outputs = self.direction_getter(dg_in)

        # -----------------------------------
        # Tracking: return raw tensor (one output per active streamline)
        # -----------------------------------
        if self.context == 'tracking':
            return model_outputs, None, None

        # -----------------------------------
        # Non-tracking: restore original structure
        # -----------------------------------
        if point_idx is not None:
            if 'gaussian' in self.dg_key or 'fisher' in self.dg_key:
                x1, x2 = model_outputs
                model_outputs = (
                    [x1[i].unsqueeze(0) for i in unsorted_indices],
                    [x2[i].unsqueeze(0) for i in unsorted_indices]
                )
            else:
                model_outputs = [
                    model_outputs[i].unsqueeze(0) for i in unsorted_indices
                ]
        else:
            if 'gaussian' in self.dg_key or 'fisher' in self.dg_key:
                x1, x2 = model_outputs

                x1 = PackedSequence(
                    x1, batch_sizes,
                    x_packed.sorted_indices, x_packed.unsorted_indices
                )
                x2 = PackedSequence(
                    x2, batch_sizes,
                    x_packed.sorted_indices, x_packed.unsorted_indices
                )

                x1 = faster_unpack_sequence(x1)
                x2 = faster_unpack_sequence(x2)

                model_outputs = (
                    [x1[i] for i in unsorted_indices],
                    [x2[i] for i in unsorted_indices]
                )
            else:
                model_outputs = PackedSequence(
                    model_outputs,
                    batch_sizes,
                    x_packed.sorted_indices,
                    x_packed.unsorted_indices
                )
                model_outputs = faster_unpack_sequence(model_outputs)
                model_outputs = [model_outputs[i] for i in unsorted_indices]

        return model_outputs, None, None