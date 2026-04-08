# -*- coding: utf-8 -*-
"""
Trainer for the TCN-GAT model.

Unlike Learn2Track with RNNs, the TCN-GAT model does not maintain a recurrent
hidden state during tracking. Therefore, at each propagation step, the model
uses the full streamline history to predict the next direction.
"""

import logging
from typing import List

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from dwi_ml.models.projects.tcn_learn2track_model import (
    TCNLearn2TrackModel,
)
from dwi_ml.tracking.io_utils import prepare_tracking_mask
from dwi_ml.tracking.propagation import propagate_multiple_lines
from dwi_ml.training.trainers_withGV import DWIMLTrainerOneInputWithGVPhase

logger = logging.getLogger("trainer_logger")


class TCNLearn2TrackTrainer(DWIMLTrainerOneInputWithGVPhase):
    """
    Trainer for the TCN-GAT model.

    The tracking validation phase uses streamline propagation based on the full
    point history because the TCN does not use recurrent hidden states.
    """
    model: TCNLearn2TrackModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def propagate_multiple_lines(
        self,
        lines: List[torch.Tensor],
        ids_per_subj
    ):
        """
        Propagate streamlines for multiple subjects.

        Parameters
        ----------
        lines : List[torch.Tensor]
            Initial streamlines/seeds to propagate.
        ids_per_subj : dict
            Mapping from subject index to slice of lines belonging to that
            subject.

        Returns
        -------
        final_lines : List[torch.Tensor]
            Propagated streamlines.
        """
        assert self.model.step_size is not None, \
            "Cannot propagate compressed streamlines."

        theta = 2 * np.pi
        max_nbr_pts = int(200 / self.model.step_size)

        final_lines = []

        for subj_idx, subj_line_idx_slice in ids_per_subj.items():
            subj_lines = lines[subj_line_idx_slice]
            subj_lines = [line for line in subj_lines if line is not None]

            if len(subj_lines) == 0:
                logger.debug("No lines to propagate for subject %s.", subj_idx)
                continue

            # Normalize streamline format
            normalized_lines = []
            for line in subj_lines:
                if not isinstance(line, torch.Tensor):
                    line = torch.as_tensor(
                        line, dtype=torch.float32, device=self.device
                    )
                else:
                    line = line.to(self.device, dtype=torch.float32)

                if line.dim() == 1:
                    line = line.unsqueeze(0)

                if line.dim() != 2 or line.shape[-1] != 3:
                    raise ValueError(
                        f"Unexpected streamline shape for subject {subj_idx}: "
                        f"{tuple(line.shape)}"
                    )

                normalized_lines.append(line)

            subj_lines = normalized_lines

            # Load subject tracking mask
            with h5py.File(self.batch_loader.dataset.hdf5_file, "r") as hdf_handle:
                subj_id = self.batch_loader.context_subset.subjects[subj_idx]
                logger.debug(
                    "Loading subject %s (%s) tracking mask.",
                    subj_idx,
                    subj_id,
                )
                tracking_mask, _ = prepare_tracking_mask(
                    hdf_handle,
                    self.tracking_mask_group,
                    subj_id=subj_id,
                    mask_interp="nearest",
                )
                tracking_mask.move_to(self.device)

            def update_memory_after_removing_lines(can_continue: np.ndarray, _):
                """
                No recurrent hidden state to update for TCN.
                """
                return

            
           
            def get_dirs_at_last_pos(current_lines: List[torch.Tensor], last_pos=None):
                nonlocal subj_idx
        
                context_len = getattr(self.model, "context_len", None)
                assert isinstance(context_len, int) and context_len > 0, \
                    "Model.context_len must be a positive integer for TCN tracking."

                hist_lines = [
                    line[-context_len:] if len(line) > context_len else line
                    for line in current_lines
                ]

                assert all(len(line) > 0 for line in hist_lines), \
                    "All streamlines must be non-empty."

                subj_dict = {subj_idx: slice(0, len(hist_lines))}
                subj_inputs = self.batch_loader.load_batch_inputs(hist_lines, subj_dict)

                with torch.no_grad():
                    flat_outputs, _, _ = self.model(
                        subj_inputs,
                        input_streamlines=hist_lines
                    )

                    lengths = torch.tensor(
                        [len(line) for line in hist_lines],
                        device=flat_outputs.device,
                        dtype=torch.long
                    )

                    expected_n = int(lengths.sum().item())
                    assert flat_outputs.shape[0] == expected_n, \
                        f"Expected {expected_n} outputs, got {flat_outputs.shape[0]}."

                    last_indices = torch.cumsum(lengths, dim=0) - 1
                    last_outputs = flat_outputs[last_indices]

                    next_dirs = self.model.get_tracking_directions(
                        last_outputs,
                        algo="det",
                        eos_stopping_thresh=0.5
                    )

                return next_dirs                                                  
            propagated_lines = propagate_multiple_lines(
                subj_lines,
                update_memory_after_removing_lines=update_memory_after_removing_lines,
                get_next_dirs=get_dirs_at_last_pos,
                theta=theta,
                step_size=self.model.step_size,
                verify_opposite_direction=False,
                mask=tracking_mask,
                max_nbr_pts=max_nbr_pts,
                append_last_point=False,
                normalize_directions=True,
            )

            for line in propagated_lines:
                if isinstance(line, np.ndarray):
                    line = torch.as_tensor(
                        line, dtype=torch.float32, device=self.device
                    )
                elif isinstance(line, torch.Tensor):
                    line = line.to(self.device, dtype=torch.float32)
                else:
                    line = torch.as_tensor(
                        line, dtype=torch.float32, device=self.device
                    )

                final_lines.append(line)

        return final_lines