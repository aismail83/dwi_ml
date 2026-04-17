# -*- coding: utf-8 -*-
import logging
import torch

from dwi_ml.tracking.tracker import DWIMLTrackerOneInput
from dwi_ml.tracking.tracking_mask import TrackingMask
from dwi_ml.models.projects.tcn_learn2track_model import (
    TCNLearn2TrackModel,
)

logger = logging.getLogger("tracker")


class TCNTracker(DWIMLTrackerOneInput):
    def __init__(
        self,
        input_volume_group,
        dataset,
        subj_idx: int,
        model: TCNLearn2TrackModel,
        mask: TrackingMask,
        seed_generator,
        nbr_seeds: int,
        min_len_mm: float,
        max_len_mm: float,
        compression_th: float,
        nbr_processes: int,
        save_seeds: bool,
        rng_seed: int,
        track_forward_only: bool,
        step_size_mm: float,
        algo: str,
        theta: float,
        use_gpu: bool,
        eos_stopping_thresh: float,
        simultaneous_tracking,
        append_last_point: bool,
        log_level=logging.INFO,
    ):
        if not isinstance(model, TCNLearn2TrackModel):
            raise TypeError(
                "TCNGATTracker requires a TCNGATLearn2TrackModel instance."
            )

        super().__init__(
            dataset=dataset,
            input_volume_group=input_volume_group,
            subj_idx=subj_idx,
            model=model,
            mask=mask,
            seed_generator=seed_generator,
            nbr_seeds=nbr_seeds,
            min_len_mm=min_len_mm,
            max_len_mm=max_len_mm,
            step_size_mm=step_size_mm,
            algo=algo,
            theta=theta,
            verify_opposite_direction=False,
            compression_th=compression_th,
            nbr_processes=nbr_processes,
            save_seeds=save_seeds,
            rng_seed=rng_seed,
            track_forward_only=track_forward_only,
            simultaneous_tracking=simultaneous_tracking,
            use_gpu=use_gpu,
            append_last_point=append_last_point,
            eos_stopping_thresh=eos_stopping_thresh,
            log_level=log_level,
        )

        self.model.set_context("tracking")
        self.input_volume_group = input_volume_group
        self.log_level = log_level

    def _to_batch_tensor(self, x):
        """
        Normalize model outputs to shape (batch, feat) whenever possible.
        """
        if isinstance(x, list):
            elems = []
            for t in x:
                if not isinstance(t, torch.Tensor):
                    t = torch.as_tensor(t, device=self.device)
                else:
                    t = t.to(self.device)

                if t.dim() == 2:
                    # (seq, feat) -> keep last timestep
                    elems.append(t[-1])
                elif t.dim() == 1:
                    elems.append(t)
                else:
                    raise ValueError(
                        f"Unexpected tensor in model_outputs list: shape={tuple(t.shape)}"
                    )
            return torch.stack(elems, dim=0)

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, device=self.device)
        else:
            x = x.to(self.device)

        if x.dim() == 3:
            # (batch, seq, feat) -> last timestep
            return x[:, -1, :]
        elif x.dim() == 2:
            return x
        elif x.dim() == 1:
            return x.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected model output shape: {tuple(x.shape)}")
    def get_dirs_at_last_pos(self, subj_lines: list[torch.Tensor], n_last_pos):
            subj_idx =self.subj_idx

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

    