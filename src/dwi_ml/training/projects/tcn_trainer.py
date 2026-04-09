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
    def propagate_multiple_lines(self, lines: List[torch.Tensor], ids_per_subj):
        """
        Tractography propagation of 'lines'.
        As compared to super, model requires an additional hidden state.
        """
        assert self.model.step_size is not None, \
            "We can't propagate compressed streamlines."

        # Setting our own limits here.
        theta = 2 * np.pi  # theta = 360 degrees
        max_nbr_pts = int(200 / self.model.step_size)

        # These methods will be used during the loop on subjects
        # Based on the tracker
        def update_memory_after_removing_lines(can_continue: np.ndarray, _):
                """
                No recurrent hidden state to update for TCN.
                """
                return
        def get_dirs_at_last_pos(subj_lines: List[torch.Tensor], n_last_pos):
            # Get dirs for current subject: run model
            nonlocal subj_idx
            
            n_last_pos = [pos[None, :] for pos in n_last_pos]
            subj_dict = {subj_idx: slice(0, len(n_last_pos))}
            subj_inputs = self.batch_loader.load_batch_inputs(n_last_pos,
                                                              subj_dict)
            model_outputs, subj_hidden_states,_ = self.model(
                subj_inputs, subj_lines)

            next_dirs = self.model.get_tracking_directions(
                model_outputs, algo='det', eos_stopping_thresh=0.5)
            
            
            return next_dirs

        # Running the propagation separately for each subject
        # (because they all need their own tracking mask)
        final_lines = []
        i = -1
        for subj_idx, subj_line_idx_slice in ids_per_subj.items():
            i += 1
            # Load the subject's tracking mask
            with h5py.File(self.batch_loader.dataset.hdf5_file, 'r'
                           ) as hdf_handle:
                subj_id = self.batch_loader.context_subset.subjects[subj_idx]
                logging.debug("Loading subj {} ({})'s tracking mask."
                              .format(subj_idx, subj_id))
                tracking_mask, _ = prepare_tracking_mask(
                    hdf_handle, self.tracking_mask_group, subj_id=subj_id,
                    mask_interp='nearest')
                tracking_mask.move_to(self.device)

            # Propagates all lines for this subject
            final_lines.extend(propagate_multiple_lines(
                lines[subj_line_idx_slice], update_memory_after_removing_lines,
                get_next_dirs=get_dirs_at_last_pos, theta=theta,
                step_size=self.model.step_size, verify_opposite_direction=False,
                mask=tracking_mask, max_nbr_pts=max_nbr_pts,
                append_last_point=False, normalize_directions=True))
        
        return final_lines

    