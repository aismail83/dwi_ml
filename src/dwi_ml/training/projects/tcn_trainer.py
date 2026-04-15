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
        Propagate streamlines for tracking validation using full streamline history.
        """
        assert self.model.step_size is not None, \
            "Propagation requires a defined step_size."

        theta =  np.pi / 3 
        max_nbr_pts = int(200 / self.model.step_size)

        def update_memory_after_removing_lines(can_continue: np.ndarray, _):
            # No recurrent state for TCN models.
            return
        
        def get_dirs_at_last_pos(subj_lines: List[torch.Tensor], n_last_pos):
            nonlocal subj_idx

            # On ignore n_last_pos comme entrée principale :
            # le TCN a besoin d'une séquence, pas d'un seul point.
            ctx = self.model.context_len

            # Fenêtre causale sur chaque streamline partielle
            ctx_lines = [
                line[-ctx:, :] if len(line) > ctx else line
                for line in subj_lines
            ]
            
            subj_dict = {subj_idx: slice(0, len(ctx_lines))}

            # Charger les features sur toute la fenêtre de contexte
            subj_inputs = self.batch_loader.load_batch_inputs(ctx_lines, subj_dict)

            # Le modèle prédit au dernier point de chaque séquence
            model_outputs, _, _ = self.model(
                subj_inputs, ctx_lines, point_idx=-1
            )

            next_dirs = self.model.get_tracking_directions(
                model_outputs, algo='det', eos_stopping_thresh=0.5
            )

            return next_dirs

        final_lines = []
        for subj_idx, subj_line_idx_slice in ids_per_subj.items():
            with h5py.File(self.batch_loader.dataset.hdf5_file, 'r') as hdf_handle:
                subj_id = self.batch_loader.context_subset.subjects[subj_idx]
                logging.debug(
                    "Loading subj %s (%s)'s tracking mask.", subj_idx, subj_id
                )
                tracking_mask, _ = prepare_tracking_mask(
                    hdf_handle, self.tracking_mask_group, subj_id=subj_id,
                    mask_interp='nearest'
                )
                tracking_mask.move_to(self.device)

            final_lines.extend(
                propagate_multiple_lines(
                    lines[subj_line_idx_slice],
                    update_memory_after_removing_lines,
                    get_next_dirs=get_dirs_at_last_pos,
                    theta=theta,
                    step_size=self.model.step_size,
                    verify_opposite_direction=False,
                    mask=tracking_mask,
                    max_nbr_pts=max_nbr_pts,
                    append_last_point=False,
                    normalize_directions=True,
                )
            )
        
        return final_lines

        
