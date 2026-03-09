# -*- coding: utf-8 -*-
import logging
from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

keys_to_rnn_class = {
    'lstm': torch.nn.LSTM,
    'gru': torch.nn.GRU
}

logger = logging.getLogger('model_logger')

ADD_SKIP_TO_OUTPUT = True


class StackedRNN(torch.nn.Module):
    """
    Recurrent model with variable recurrent layer sizes, and optional skip
    connections.
    """

    def __init__(self, rnn_torch_key: str, input_size: int,
                 layer_sizes: List[int], use_skip_connection: bool,
                 use_layer_normalization: bool, dropout: float,
                 predict_bundle_ids: bool = False,
                 num_bundles: int = 21):
        """
        Parameters
        ----------
        rnn_torch_key : str
            Choice of recurrent cell: 'lstm' or 'gru'.
        input_size : int
            Feature size at each step.
        layer_sizes : list[int]
            Hidden size of each recurrent layer.
        use_skip_connection : bool
            If True, concatenate initial input to each intermediate layer input,
            and concatenate all layer outputs at the end.
        use_layer_normalization : bool
            If True, apply layer normalization after each recurrent layer.
        dropout : float
            Dropout applied after each recurrent layer except the last.
        predict_bundle_ids : bool
            If True, add a linear head on top of final output.
        num_bundles : int
            Number of bundle classes for the optional head.
        """
        if rnn_torch_key not in keys_to_rnn_class:
            raise ValueError(f"Unsupported rnn_torch_key: {rnn_torch_key}. "
                             f"Expected one of {list(keys_to_rnn_class.keys())}")

        if not isinstance(dropout, float) or not 0 <= dropout <= 1:
            raise ValueError(
                "dropout should be a rate in range [0, 1] representing the "
                "probability of an element being zeroed"
            )

        if dropout > 0 and len(layer_sizes) == 1:
            logging.warning(
                "dropout option adds dropout after all but last recurrent "
                "layer, so non-zero dropout expects num_layers greater than 1, "
                f"but got dropout={dropout} and len(layer_sizes)={len(layer_sizes)}"
            )

        if use_skip_connection and len(layer_sizes) == 1 and not ADD_SKIP_TO_OUTPUT:
            logging.warning(
                "With only one layer, the skip connection has no effect with "
                "current architecture."
            )

        super().__init__()

        self.rnn_torch_key = rnn_torch_key
        self.input_size = int(input_size)
        self.layer_sizes = list(layer_sizes)
        self.use_skip_connection = bool(use_skip_connection)
        self.use_layer_normalization = bool(use_layer_normalization)
        self.dropout = float(dropout)

        self.predict_bundle_ids = bool(predict_bundle_ids)
        self.num_bundles = int(num_bundles)

        self.rnn_layers = []
        self.layer_norm_layers = []

        self.dropout_module = (
            torch.nn.Dropout(self.dropout) if self.dropout > 0 else None
        )
        self.relu_sublayer = torch.nn.ReLU()

        rnn_cls = keys_to_rnn_class[self.rnn_torch_key]
        last_layer_input_size = self.input_size

        # Create recurrent layers exactly once
        for i, layer_size in enumerate(self.layer_sizes):
            layer_size = int(layer_size)

            rnn_layer = rnn_cls(
                input_size=last_layer_input_size,
                hidden_size=layer_size,
                num_layers=1,
                batch_first=True
            )
            self.add_module(f"rnn_{i}", rnn_layer)
            self.rnn_layers.append(rnn_layer)

            if self.use_layer_normalization:
                layer_norm = torch.nn.LayerNorm(layer_size)
                self.add_module(f"layer_norm_{i}", layer_norm)
                self.layer_norm_layers.append(layer_norm)

            # Next layer input size
            if self.use_skip_connection:
                last_layer_input_size = layer_size + self.input_size
            else:
                last_layer_input_size = layer_size

        # Optional bundle classification head
        if self.predict_bundle_ids:
            if self.num_bundles <= 0:
                raise ValueError(
                    "num_bundles must be > 0 when predict_bundle_ids=True"
                )
            self.bundle_head = torch.nn.Linear(self.output_size, self.num_bundles)
        else:
            self.bundle_head = None

        logger.info("=== StackedRNN initialized ===")
        for i, r in enumerate(self.rnn_layers):
            logger.info(
                f"layer {i}: input_size={r.input_size}, hidden_size={r.hidden_size}"
            )
        logger.info(f"nb layers = {len(self.rnn_layers)}")
        logger.info(f"model input_size = {self.input_size}")
        logger.info(f"model output_size = {self.output_size}")

    @property
    def params(self):
        return {
            'rnn_torch_key': self.rnn_torch_key,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'layer_sizes': list(self.layer_sizes),
            'use_skip_connections': self.use_skip_connection,
            'use_layer_normalization': self.use_layer_normalization,
            'dropout': self.dropout,
        }

    @property
    def output_size(self):
        """
        Final output size of the stacked RNN.
        """
        if self.use_skip_connection:
            if ADD_SKIP_TO_OUTPUT:
                return sum(self.layer_sizes) + self.input_size
            else:
                return sum(self.layer_sizes)
        else:
            return self.layer_sizes[-1]

    def forward(self, inputs: PackedSequence,
                hidden_states: Tuple[Tensor, ...] = None):
        """
        Parameters
        ----------
        inputs : PackedSequence
        hidden_states : list[states]
            One value per layer.

        Returns
        -------
        last_output : Tensor
            PackedSequence.data-like tensor.
        out_hidden_states : list
            Final hidden states of each recurrent layer.
        """
        init_inputs = inputs.data

        if hidden_states is None:
            hidden_states = [None for _ in range(len(self.rnn_layers))]

        out_hidden_states = []
        outputs = []

        last_output = inputs
        for i in range(len(self.rnn_layers)):
            logger.debug(f'Applying StackedRNN layer #{i}: {self.rnn_layers[i]}')

            if i > 0:
                last_output = PackedSequence(
                    last_output,
                    inputs.batch_sizes,
                    inputs.sorted_indices,
                    inputs.unsorted_indices
                )

            if isinstance(last_output, PackedSequence):
                logger.debug(
                    f"[layer {i}] packed data shape = {last_output.data.shape}, "
                    f"expected input_size = {self.rnn_layers[i].input_size}"
                )
            else:
                logger.debug(
                    f"[layer {i}] tensor shape = {last_output.shape}, "
                    f"expected input_size = {self.rnn_layers[i].input_size}"
                )

            last_output, new_state_i = self.rnn_layers[i](
                last_output, hidden_states[i]
            )
            out_hidden_states.append(new_state_i)

            last_output = last_output.data

            if self.use_layer_normalization:
                last_output = self.layer_norm_layers[i](last_output)

            if i < len(self.rnn_layers) - 1:
                if self.dropout_module is not None:
                    last_output = self.dropout_module(last_output)
                    logger.debug(
                        f'   Output size after dropout: {last_output.shape}'
                    )

                last_output = self.relu_sublayer(last_output)
                logger.debug(f'   Output size after ReLU: {last_output.shape}')

            if self.use_skip_connection:
                outputs.append(last_output)

                if i < len(self.rnn_layers) - 1:
                    last_output = torch.cat((last_output, init_inputs), dim=-1)
                    logger.debug(
                        f'   Output size after skip connection: {last_output.shape}'
                    )

        if self.use_skip_connection:
            if len(self.rnn_layers) > 1:
                last_output = torch.cat(outputs, dim=-1)
            else:
                last_output = outputs[0]

            if ADD_SKIP_TO_OUTPUT:
                last_output = torch.cat((last_output, init_inputs), dim=-1)
                logger.debug(
                    'Final skip connection: concatenating all outputs AND input. '
                    f'Final shape is {last_output.shape}'
                )
            else:
                logger.debug(
                    'Final skip connection: concatenating all outputs but NOT input. '
                    f'Final shape is {last_output.shape}'
                )

        if self.bundle_head is None:
            return last_output, out_hidden_states
        else:
            bundle_logits = self.bundle_head(last_output)
            return last_output, out_hidden_states, bundle_logits