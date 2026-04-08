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

    def get_next_dirs(self, lines, n_last_pos):
        inputs = self._prepare_inputs_at_pos(n_last_pos)
        model_outputs = self._call_model_forward(inputs, lines)

        # Common case: (dir_output, None, None)
        if (
            isinstance(model_outputs, tuple)
            and len(model_outputs) == 3
            and model_outputs[1] is None
            and model_outputs[2] is None
        ):
            model_outputs = model_outputs[0]
        print(model_outputs)
        # Fisher-von-Mises-style outputs:
        # possible form: (mus, kappas, eos)
        if isinstance(model_outputs, (tuple, list)) and len(model_outputs) >= 2:
            mus = self._to_batch_tensor(model_outputs[0])
            kappas = self._to_batch_tensor(model_outputs[1])
            model_outputs = (mus, kappas)

        elif isinstance(model_outputs, list):
            model_outputs = self._to_batch_tensor(model_outputs)

        elif isinstance(model_outputs, torch.Tensor):
            model_outputs = self._to_batch_tensor(model_outputs)

        next_dirs = self.model.get_tracking_directions(
            model_outputs, self.algo, self.eos_stopping_thresh
        )
        return next_dirs