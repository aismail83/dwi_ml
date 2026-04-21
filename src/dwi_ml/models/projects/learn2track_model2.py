# -*- coding: utf-8 -*-
import logging
from typing import Union, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import invert_permutation, PackedSequence, pack_sequence

from dwi_ml.data.processing.space.neighborhood import unflatten_neighborhood
from dwi_ml.data.processing.streamlines.post_processing import \
    compute_directions, normalize_directions, compute_n_previous_dirs
from dwi_ml.data.processing.streamlines.sos_eos_management import \
    convert_dirs_to_class
from dwi_ml.models.embeddings import NoEmbedding
from dwi_ml.models.main_models import (
    ModelWithPreviousDirections, ModelWithDirectionGetter,
    ModelWithNeighborhood, ModelWithOneInput)
from dwi_ml.models.stacked_rnn import StackedRNN

logger = logging.getLogger('model_logger')  # Same logger as Super.


def faster_unpack_sequence(packed_sequence: PackedSequence):
    # To be used with ordered batch
    # From my tests, seems to be ~twice faster than
    # torch.nn.utils.rnn.unpack_sequence

    # Note:
    # len(batch_size) = max number of points.
    # sum(batch_size) = total number of points
    # values in batch_size: ex, [1000, 1000, 3], means that the first
    # 1000 points are separate lines (we understand that we have 1000 lines).
    # Then the next 1000 points also are separated lines. Then last 3 points
    # are separated lines. So out we have 997 lines with 2 points and
    # 3 with 3 points.
    nb_lines = packed_sequence.batch_sizes[0]

    # Indices of points of the first line.
    ind = [-1]
    for nb_pts in packed_sequence.batch_sizes[:-1]:
        ind.append(ind[-1] + nb_pts)
    ind = np.asarray(ind)

    batch = []
    count_nb_lines_this_size = 0
    remaining_batch_sizes = packed_sequence.batch_sizes.detach().clone()
    total_nb_lines_this_size = remaining_batch_sizes[-1]
    for i in range(nb_lines):
        count_nb_lines_this_size += 1

        if count_nb_lines_this_size > total_nb_lines_this_size:
            # Done for lines of this size. Next size:
            previous_nb = remaining_batch_sizes[-1]
            while remaining_batch_sizes[-1] == previous_nb:
                ind = ind[:-1]
                remaining_batch_sizes = remaining_batch_sizes[:-1]

            count_nb_lines_this_size = 1
            total_nb_lines_this_size = remaining_batch_sizes[-1] - previous_nb

        ind += 1
        line = packed_sequence.data[ind]
        # Note. len(line) == len(remaining_batch_sizes)
        batch.append(line)

    return batch


class Learn2TrackModel(ModelWithPreviousDirections, ModelWithDirectionGetter,
                       ModelWithNeighborhood, ModelWithOneInput):
    """
    Recurrent tracking model.

    Composed of an embedding for the imaging data's input + for the previous
    direction's input, an RNN model to process the sequences, and a direction
    getter model to convert the RNN outputs to the right structure, e.g.
    deterministic (3D vectors) or probabilistic (based on probability
    distribution parameters).
    """

    def __init__(self, experiment_name,
                 step_size: Union[float, None],
                 compress_lines: Union[float, None],
                 nb_features: int,
                 # PREVIOUS DIRS
                 nb_previous_dirs: Union[int, None],
                 prev_dirs_embedded_size: Union[int, None],
                 prev_dirs_embedding_key: Union[str, None],
                 normalize_prev_dirs: bool,
                 # INPUTS
                 input_embedding_key: str,
                 input_embedded_size: Union[int, None],
                 nb_cnn_filters: Optional[List[int]],
                 kernel_size: Optional[List[int]],
                 # RNN
                 rnn_key: str, rnn_layer_sizes: List[int],
                 use_skip_connection: bool, use_layer_normalization: bool,
                 dropout: float,
                 # DIRECTION GETTER
                 dg_key: str, dg_args: Union[dict, None],
                 # Other
                 neighborhood_type: Optional[str] = None,
                 neighborhood_radius: Optional[int] = None,
                 neighborhood_resolution: Optional[float] = None,
                 log_level=logging.root.level,
                 nb_points: Optional[int] = None,
                 # Bundle options (MUST be in init for checkpoint reload)
                 use_bundle_ids: bool = False,
                 predict_bundle_ids: bool = False,
                 bundle_emb_dim: Optional[int] = None,
                 num_bundles: Optional[int] = None):
        """
        Params
        ------
        nb_previous_dirs: int
            Number of previous direction (i.e. [x,y,z] information) to be
            received.
        rnn_key: str
            Either 'LSTM' or 'GRU'.
        rnn_layer_sizes: List[int]
            The list of layer sizes for the rnn. The real size will depend
            on the skip_connection parameter.
        use_skip_connection: bool
            Whether to use skip connections. See [1] (Figure 1) to visualize
            the architecture.
        use_layer_normalization: bool
            Whether to apply layer normalization to the forward connections.
            See [2].
        dropout : float
            If non-zero, introduces a `Dropout` layer on the outputs of each
            RNN layer except the last layer, with given dropout probability.
        ---
        [1] https://arxiv.org/pdf/1308.0850v5.pdf
        [2] https://arxiv.org/pdf/1607.06450.pdf
        """
        super().__init__(
            experiment_name=experiment_name, step_size=step_size,
            nb_points=nb_points,
            compress_lines=compress_lines, log_level=log_level,
            # For modelWithNeighborhood
            neighborhood_type=neighborhood_type,
            neighborhood_radius=neighborhood_radius,
            neighborhood_resolution=neighborhood_resolution,
            # For super ModelWithInputEmbedding:
            nb_features=nb_features,
            input_embedding_key=input_embedding_key,
            input_embedded_size=input_embedded_size,
            nb_cnn_filters=nb_cnn_filters, kernel_size=kernel_size,
            # For super MainModelWithPD:
            nb_previous_dirs=nb_previous_dirs,
            prev_dirs_embedded_size=prev_dirs_embedded_size,
            prev_dirs_embedding_key=prev_dirs_embedding_key,
            normalize_prev_dirs=normalize_prev_dirs,
            # For super ModelForTracking:
            dg_args=dg_args, dg_key=dg_key)

        self.dropout = dropout
        self.nb_cnn_filters = nb_cnn_filters
        self.kernel_size = kernel_size
        self.good_indices=None
        if dropout < 0 or dropout > 1:
            raise ValueError('The dropout rate must be between 0 and 1.')

        # ---- Bundle ID options
        self.use_bundle_ids = bool(use_bundle_ids)
        self.predict_bundle_ids = bool(predict_bundle_ids)
        self.num_bundles = int(num_bundles) if num_bundles is not None else 0

        if self.use_bundle_ids:
            if bundle_emb_dim is None:
                raise ValueError(
                    "bundle_emb_dim must be provided when use_bundle_ids=True")
            if num_bundles is None:
                raise ValueError(
                    "num_bundles must be provided when use_bundle_ids=True")

            self.bundle_emb_dim = int(bundle_emb_dim)

            if self.bundle_emb_dim <= 0:
                raise ValueError(
                    f"bundle_emb_dim must be > 0 "
                    f"(got {self.bundle_emb_dim})")
            if self.num_bundles <= 0:
                raise ValueError(
                    f"num_bundles must be > 0 (got {self.num_bundles})")

            self.bundle_emb = torch.nn.Embedding(
                self.num_bundles, self.bundle_emb_dim)
        else:
            self.bundle_emb_dim = 0
            self.bundle_emb = None

        # Raw size before input embedding.
        self.raw_input_size = nb_features * self.nb_neighbors

        # Size entering the RNN.
        self.input_size = self.computed_input_embedded_size

        # + bundle embedding only if enabled
        if self.use_bundle_ids:
            self.input_size += self.bundle_emb_dim

        # + previous dirs embedding
        if self.nb_previous_dirs > 0:
            self.input_size += self.prev_dirs_embedded_size

        # ---------- Instantiations
        # 1. Previous dirs embedding: prepared by super.

        # 2. Input embedding
        self.embedding_dropout = torch.nn.Dropout(self.dropout)

        # 3. Stacked RNN
        if len(rnn_layer_sizes) == 1:
            dropout = 0.0  # Not used in RNN. Avoiding the warning.

        self.rnn_model = StackedRNN(
            rnn_key, self.input_size, rnn_layer_sizes,
            use_skip_connection=use_skip_connection,
            use_layer_normalization=use_layer_normalization,
            dropout=dropout,
            predict_bundle_ids=self.predict_bundle_ids,
            num_bundles=self.num_bundles
        )

        # 4. Direction getter
        self.instantiate_direction_getter(self.rnn_model.output_size)

    def set_context(self, context):
        # Training, validation: Used by trainer. Nothing special.
        # Tracking: Used by tracker. Returns only the last point.
        #     Preparing_backward: Used by tracker. Nothing special, but does
        #     not return only the last point.
        # Visu: Nothing special. Used by tester.
        assert context in ['training', 'validation', 'tracking', 'visu',
                           'preparing_backward']
        self._context = context

    @property
    def params_for_checkpoint(self):
        # Every parameter necessary to build the different layers again
        # during checkpoint state saving.
        params = super().params_for_checkpoint
        params.update({
            'nb_features': int(self.nb_features),
            'rnn_key': self.rnn_model.rnn_torch_key,
            'rnn_layer_sizes': self.rnn_model.layer_sizes,
            'use_skip_connection': self.rnn_model.use_skip_connection,
            'use_layer_normalization': self.rnn_model.use_layer_normalization,
            'dropout': self.dropout,
            'use_bundle_ids': self.use_bundle_ids,
            'bundle_emb_dim': self.bundle_emb_dim,
            'num_bundles': self.num_bundles,
            'predict_bundle_ids': self.predict_bundle_ids
        })

        return params

    @property
    def computed_params_for_display(self):
        p = super().computed_params_for_display
        p['stacked_RNN_output_size'] = self.rnn_model.output_size
        return p

    def forward(self, x: List[torch.Tensor],
                input_streamlines: List[torch.Tensor] = None,
                bundle_ids: torch.Tensor = None,
                hidden_recurrent_states: List = None,
                return_hidden: bool = False,
                point_idx: int = None):
        """Run the model on a batch of sequences."""

        if self.context is None:
            raise ValueError("Please set context before usage.")

        assert x[0].shape[-1] == self.raw_input_size, \
            "Not the expected input size! Should be {} (i.e. {} features for " \
            "each of the {} neighbors), but got {} (input shape {})." \
            .format(self.raw_input_size, self.nb_features, self.nb_neighbors,
                    x[0].shape[-1], x[0].shape)

        self.good_indices = None

        unsorted_indices = None
        sorted_indices = None
        if self.context != 'tracking':
            lengths = torch.as_tensor([len(s) for s in x])
            _, sorted_indices = torch.sort(lengths, descending=True)
            unsorted_indices = invert_permutation(sorted_indices)

            x = [x[i] for i in sorted_indices]
            if input_streamlines is not None:
                input_streamlines = [input_streamlines[i] for i in sorted_indices]

        dev = next(self.parameters()).device

        # -------------------------
        # Bundle ids
        # -------------------------
        if self.use_bundle_ids:
            if self.context == 'tracking' or bundle_ids is None:
                bundle_ids = torch.zeros(len(x), device=dev, dtype=torch.long)
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

        # -------------------------
        # Previous directions
        # -------------------------
        n_prev_dirs = None
        if self.nb_previous_dirs > 0:
            if input_streamlines is None:
                raise ValueError(
                    "input_streamlines must be provided when nb_previous_dirs > 0"
                )

            dirs = compute_directions(input_streamlines)
            if self.normalize_prev_dirs:
                dirs = normalize_directions(dirs)

            n_prev_dirs = compute_n_previous_dirs(
                dirs, self.nb_previous_dirs, point_idx=point_idx
            )
            n_prev_dirs = pack_sequence(n_prev_dirs, enforce_sorted=False)
            n_prev_dirs = self.prev_dirs_embedding(n_prev_dirs.data)
            n_prev_dirs = self.embedding_dropout(n_prev_dirs)

        # -------------------------
        # Pack inputs
        # -------------------------
        x = pack_sequence(x)

        # -------------------------
        # Input embedding
        # -------------------------
        if self.nb_previous_dirs > 0 or not isinstance(self.input_embedding_layer, NoEmbedding):
            x_data = x.data

            if self.input_embedding_key == 'cnn_embedding':
                x_data = unflatten_neighborhood(
                    x_data,
                    self.neighborhood_vectors,
                    self.neighborhood_type,
                    self.neighborhood_radius,
                    self.neighborhood_resolution
                )

            x_data = self.input_embedding_layer(x_data)
            x_data = self.embedding_dropout(x_data)

            if self.nb_previous_dirs > 0:
                x_data = torch.cat((x_data, n_prev_dirs), dim=-1)

            x = PackedSequence(
                x_data,
                x.batch_sizes,
                x.sorted_indices,
                x.unsorted_indices
            )

        # -------------------------
        # RNN / main sequence model
        # -------------------------
        final_ref = x
        bundle_logits_per_line = None

        def _filter_hidden(h, idx):
            if h is None:
                return None
            if isinstance(h, tuple):  # LSTM
                return tuple(v.index_select(1, idx) for v in h)
            return h.index_select(1, idx)

        if self.predict_bundle_ids:
            if self.context != "tracking":
                x_raw = x

                # First pass
                x_first = x_raw
                if self.use_bundle_ids and \
                x_raw.data.shape[-1] == self.rnn_model.input_size - self.bundle_emb_dim:
                    x_raw_seq = faster_unpack_sequence(x_raw)

                    x_first_seq = []
                    for seq in x_raw_seq:
                        zero_seq = torch.zeros(
                            seq.shape[0],
                            self.bundle_emb_dim,
                            device=seq.device,
                            dtype=seq.dtype
                        )
                        x_first_seq.append(torch.cat((seq, zero_seq), dim=-1))

                    x_first = pack_sequence(x_first_seq)

                x, out_hidden_recurrent_states, bundle_logits_per_point = \
                    self.rnn_model(x_first, hidden_recurrent_states)

                final_ref = x_first

                # Per-line bundle logits
                bl = PackedSequence(
                    bundle_logits_per_point,
                    x_first.batch_sizes,
                    x_first.sorted_indices,
                    x_first.unsorted_indices
                )
                bl_list = faster_unpack_sequence(bl)
                bundle_logits_per_line = torch.vstack([t[-1] for t in bl_list])
                batch_pred = torch.argmax(bundle_logits_per_line, dim=1)

                # Second pass on correctly predicted streamlines
                good_indices_sorted = torch.where(batch_pred == bundle_ids)[0]
                if bundle_ids is not None and good_indices_sorted.numel() != 0:
                    
                    
                    # Store ORIGINAL indices for run_one_batch / targets filtering
                    if sorted_indices is not None:
                        self.good_indices = sorted_indices.to(good_indices_sorted.device)[good_indices_sorted]
                    else:
                        self.good_indices = good_indices_sorted
                        
                    if self.use_bundle_ids :
                        x_raw_seq = faster_unpack_sequence(x_raw)
                        x_raw_seq = [x_raw_seq[i] for i in good_indices_sorted.tolist()]

                        bundle_ids_good = bundle_ids[good_indices_sorted]
                        b = self.bundle_emb(bundle_ids_good)

                        x_second_seq = []
                        for i, seq in enumerate(x_raw_seq):
                            b_seq = b[i].expand(seq.shape[0], -1)
                            x_second_seq.append(torch.cat((seq, b_seq), dim=-1))

                        x_second = pack_sequence(x_second_seq)
                        hidden_second = _filter_hidden(hidden_recurrent_states, good_indices_sorted)

                        x, out_hidden_recurrent_states, bundle_logits_per_point = \
                            self.rnn_model(x_second, hidden_second)

                        final_ref = x_second
                else:
                    if self.use_bundle_ids and x_raw.data.shape[-1] == self.rnn_model.input_size - self.bundle_emb_dim:
                        zero_bundle = torch.zeros(
                            x_raw.data.shape[0],
                            self.bundle_emb_dim,
                            device=x_raw.data.device,
                            dtype=x_raw.data.dtype
                        )
                        x_raw = PackedSequence(
                            torch.cat((x_raw.data, zero_bundle), dim=-1),
                            x_raw.batch_sizes,
                            x_raw.sorted_indices,
                            x_raw.unsorted_indices
                        )

                    x, out_hidden_recurrent_states, bundle_logits_per_line = self.rnn_model(
                        x_raw, hidden_recurrent_states
                    )
                    final_ref = x_raw
                    self.good_indices = torch.arange(len(bundle_ids), device=bundle_ids.device)

            else:
                if self.use_bundle_ids and x.data.shape[-1] == self.rnn_model.input_size - self.bundle_emb_dim:
                    zero_bundle = torch.zeros(
                        x.data.shape[0],
                        self.bundle_emb_dim,
                        device=x.data.device,
                        dtype=x.data.dtype
                    )
                    x = PackedSequence(
                        torch.cat((x.data, zero_bundle), dim=-1),
                        x.batch_sizes,
                        x.sorted_indices,
                        x.unsorted_indices
                    )

                x, out_hidden_recurrent_states, bundle_logits_per_point = self.rnn_model(x, hidden_recurrent_states)
                bundle_logits_per_line = bundle_logits_per_point
                final_ref = x

        else:
            x, out_hidden_recurrent_states = self.rnn_model(x, hidden_recurrent_states)
            bundle_logits_per_line = None
            final_ref = x

        # -------------------------
        # Direction getter
        # -------------------------
        logger.debug("*** 5. Direction getter....")

        assert x.data.shape[-1] == self.direction_getter.input_size, \
            "Expecting input to direction getter to be of size {}. Got {}" \
            .format(self.direction_getter.input_size, x.data.shape[-1])

        x = self.direction_getter(x)

        # -------------------------
        # Unpack outputs
        # -------------------------
        if self.context != 'tracking':
            if 'gaussian' in self.dg_key or 'fisher' in self.dg_key:
                x, x2 = x

                x2 = PackedSequence(
                    x2,
                    final_ref.batch_sizes,
                    final_ref.sorted_indices,
                    final_ref.unsorted_indices
                )
                x2 = faster_unpack_sequence(x2)

            x = PackedSequence(
                x,
                final_ref.batch_sizes,
                final_ref.sorted_indices,
                final_ref.unsorted_indices
            )
            x = faster_unpack_sequence(x)

            if 'gaussian' in self.dg_key or 'fisher' in self.dg_key:
                x = (x, x2)

        # -------------------------
        # Hidden states
        # -------------------------
        if not return_hidden:
            out_hidden_recurrent_states = None
        
        return x, out_hidden_recurrent_states, bundle_logits_per_line
    
    
    def take_lines_in_hidden_state(self, hidden_states, lines_to_keep):
        """
        Utilitary method to remove a few streamlines from the hidden
        state.
        """
        if self.rnn_model.rnn_torch_key == 'lstm':
            # LSTM: For each layer, states are tuples; (h_t, C_t)
            # Size of tensors are each [1, nb_streamlines, nb_neurons]
            hidden_states = [(layer_states[0][:, lines_to_keep, :],
                              layer_states[1][:, lines_to_keep, :]) for
                             layer_states in hidden_states]
        else:
            # GRU: For each layer, states are tensors; h_t.
            # Size of tensors are [1, nb_streamlines, nb_neurons].
            hidden_states = [layer_states[:, lines_to_keep, :] for
                             layer_states in hidden_states]
        return hidden_states
