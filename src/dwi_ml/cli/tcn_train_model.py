#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a TCN-GAT model for tractography.
"""

import argparse
import gc
import inspect
import logging
import os
import warnings

# comet_ml must be imported before torch in some setups.
import comet_ml  # noqa: F401
import torch

from scilpy.io.utils import (
    add_verbose_arg,
    assert_inputs_exist,
    assert_outputs_exist,
)

from dwi_ml.data.dataset.utils import prepare_multisubjectdataset
from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.io_utils import add_memory_args

from dwi_ml.models.main_models import MainModelAbstract
from dwi_ml.models.projects.learn2track_utils import add_model_args
from dwi_ml.models.projects.tcn_learn2track_model import (
    TCNLearn2TrackModel
)
from dwi_ml.models.utils.direction_getters import check_args_direction_getter

from dwi_ml.training.projects.tcn_trainer import (
    TCNLearn2TrackTrainer
)
from dwi_ml.training.utils.batch_loaders import (
    add_args_batch_loader,
    prepare_batch_loader,
)
from dwi_ml.training.utils.batch_samplers import (
    add_args_batch_sampler,
    prepare_batch_sampler,
)
from dwi_ml.training.utils.experiment import (
    add_mandatory_args_experiment_and_hdf5_path,
)
from dwi_ml.training.utils.trainer import (
    add_training_args,
    format_lr,
    run_experiment,
)

import dwi_ml.experiment_utils.memory as memory_utils
import dwi_ml.training.trainers as trainers_module


warnings.filterwarnings(
    "ignore",
    message="`torch.distributed.reduce_op` is deprecated"
)


def _safe_log_gpu_per_tensor(*args, **kwargs):
    """
    Safe replacement for log_gpu_per_tensor.

    Compatible with any calling signature used in the codebase.
    Logs CUDA tensors without crashing if an object is malformed.
    """
    logger = kwargs.get("logger", None)

    if logger is None and len(args) > 0:
        logger = args[0]

    for obj in gc.get_objects():
        try:
            tensor = None

            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                tensor = obj.data
            elif hasattr(obj, "tensor") and torch.is_tensor(obj.tensor):
                tensor = obj.tensor

            if tensor is not None and tensor.is_cuda and logger is not None:
                logger.debug(
                    "GPU tensor: type=%s shape=%s dtype=%s device=%s",
                    type(obj).__name__,
                    tuple(tensor.shape),
                    tensor.dtype,
                    tensor.device,
                )

        except Exception:
            continue


# Monkey patch safer GPU logging
memory_utils.log_gpu_per_tensor = _safe_log_gpu_per_tensor
trainers_module.log_gpu_per_tensor = _safe_log_gpu_per_tensor


def add_model_args_tcn_gat(parser):
    """
    Add command-line arguments specific to the TCN-GAT model.
    """
    parser.add_argument(
        "--tcn_hidden_size",
        type=int,
        default=32,
        help="Hidden channels in the TCN.",
    )
    parser.add_argument(
        "--tcn_num_layers",
        type=int,
        default=2,
        help="Number of TCN layers.",
    )
    parser.add_argument(
        "--tcn_kernel_size",
        type=int,
        default=3,
        help="Kernel size for TCN convolutions.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1,
        help="Number of output classes (1 for binary classification).",
    )
    return parser


def prepare_arg_parser():
    """
    Build and return the main argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Train a TCN-GAT model for tractography."
    )

    add_mandatory_args_experiment_and_hdf5_path(parser)
    add_args_batch_sampler(parser)
    add_args_batch_loader(parser)
    add_training_args(parser, add_a_tracking_validation_phase=True)
    add_memory_args(parser, add_lazy_options=True, add_rng=True)
    add_verbose_arg(parser)

    # Base Learn2Track arguments
    add_model_args(parser)

    # TCN-GAT specific arguments
    add_model_args_tcn_gat(parser)

    return parser


def _safe_getattr(obj, name, default=None):
    """
    Safe getattr wrapper.
    """
    return getattr(obj, name, default)


def init_from_args(args, sub_loggers_level):
    """
    Build dataset, model, batch sampler, batch loader, and trainer
    from parsed command-line arguments.
    """
    torch.manual_seed(args.rng)

    dataset = prepare_multisubjectdataset(
        args,
        load_testing=False,
        log_level=sub_loggers_level,
    )

    # Temporary abstract model for loader initialization
    model_abstract = MainModelAbstract(
        experiment_name=args.experiment_name,
        subset=dataset.training_set,
        step_size=args.step_size,
        compress_lines=args.compress_th,
    )

    group_loader = prepare_batch_loader(
        dataset,
        model_abstract,
        args,
        sub_loggers_level,
    )

    check_args_direction_getter(args)

    input_group_idx = dataset.volume_groups.index(args.input_group_name)
    args.nb_features = dataset.nb_features[input_group_idx]
    dg_args = check_args_direction_getter(args)

    with Timer("\n\nPreparing TCN-GAT model", newline=True, color="yellow"):
        

        model = TCNLearn2TrackModel(
            experiment_name=args.experiment_name,
            step_size=args.step_size,
            compress_lines=args.compress_th,

            # Previous directions
            prev_dirs_embedding_key=args.prev_dirs_embedding_key,
            prev_dirs_embedded_size=args.prev_dirs_embedded_size,
            nb_previous_dirs=args.nb_previous_dirs,
            normalize_prev_dirs=args.normalize_prev_dirs,

            # Input embedding
            input_embedding_key=args.input_embedding_key,
            input_embedded_size=args.input_embedded_size,
            nb_features=args.nb_features,
            kernel_size=args.kernel_size,
            nb_cnn_filters=args.nb_cnn_filters,

            # TCN
            tcn_hidden_size=args.tcn_hidden_size,
            tcn_num_layers=args.tcn_num_layers,
            tcn_kernel_size=args.tcn_kernel_size,

            # Direction getter
            dg_key=args.dg_key,
            dg_args=dg_args,
            # Neighborhood
            neighborhood_type=args.neighborhood_type,
            neighborhood_radius=args.neighborhood_radius,
            neighborhood_resolution=args.neighborhood_resolution,

            # Bundle options
            use_bundle_ids=_safe_getattr(group_loader, "use_bundle_ids", False),
            bundle_emb_dim=_safe_getattr(group_loader, "bundle_emb_dim", None),
            num_bundles=args.num_bundles,
            predict_bundle_ids=_safe_getattr(
                group_loader, "predict_bundle_ids", False
            ),

            # Misc
            dropout=args.dropout,
            log_level=sub_loggers_level,
        )

        logging.info(
            "Imported TCNGATLearn2TrackModel from: %s",
            inspect.getfile(TCNLearn2TrackModel),
        )

        logging.info(
            "Class has params_for_checkpoint: %s",
            hasattr(TCNLearn2TrackModel, "params_for_checkpoint"),
        )

        logging.info(
            "Instance has params_for_checkpoint: %s",
            hasattr(model, "params_for_checkpoint"),
        )

        if hasattr(model, "params_for_checkpoint"):
            logging.info(
                "TCN-GAT model final parameters: %s",
                format_dict_to_str(model.params_for_checkpoint),
            )
        else:
            logging.warning("Model has no params_for_checkpoint attribute.")

        if hasattr(model, "computed_params_for_display"):
            logging.info(
                "Computed parameters: %s",
                format_dict_to_str(model.computed_params_for_display),
            )

    batch_sampler = prepare_batch_sampler(dataset, args, sub_loggers_level)
    batch_loader = prepare_batch_loader(dataset, model, args, sub_loggers_level)

    with Timer("\n\nPreparing trainer", newline=True, color="red"):
        learning_rates = format_lr(args.learning_rate)

        trainer = TCNLearn2TrackTrainer(
            model=model,
            experiments_path=args.experiments_path,
            experiment_name=args.experiment_name,
            batch_sampler=batch_sampler,
            batch_loader=batch_loader,

            # Comet logging
            comet_project=args.comet_project,
            comet_workspace=args.comet_workspace,

            # Optimization
            learning_rates=learning_rates,
            weight_decay=args.weight_decay,
            optimizer=args.optimizer,
            max_epochs=args.max_epochs,
            max_batches_per_epoch_training=args.max_batches_per_epoch_training,
            max_batches_per_epoch_validation=args.max_batches_per_epoch_validation,
            patience=args.patience,
            patience_delta=args.patience_delta,
            from_checkpoint=False,
            clip_grad=args.clip_grad,

            # Validation / tracking
            add_a_tracking_validation_phase=args.add_a_tracking_validation_phase,
            tracking_phase_frequency=args.tracking_phase_frequency,
            tracking_phase_nb_segments_init=args.tracking_phase_nb_segments_init,
            tracking_phase_mask_group=args.tracking_mask,

            # Compute
            nb_cpu_processes=args.nbr_processes,
            use_gpu=args.use_gpu,
            log_level=args.verbose,
        )

        if hasattr(trainer, "params_for_checkpoint"):
            logging.info(
                "Trainer params: %s",
                format_dict_to_str(trainer.params_for_checkpoint),
            )

    return trainer


def main():
    """
    Main entry point.
    """
    parser = prepare_arg_parser()
    args = parser.parse_args()

    sub_loggers_level = args.verbose if args.verbose != "DEBUG" else "INFO"

    logging.getLogger().setLevel(logging.WARNING)

    assert_inputs_exist(parser, [args.hdf5_file])
    assert_outputs_exist(parser, args, args.experiments_path)

    checkpoint_dir = os.path.join(
        args.experiments_path,
        args.experiment_name,
        "checkpoint",
    )

    if os.path.exists(checkpoint_dir):
        raise FileExistsError(
            "This experiment already exists. Delete it or use "
            "l2t_resume_training_from_checkpoint.py."
        )

    trainer = init_from_args(args, sub_loggers_level)
    run_experiment(trainer)


if __name__ == "__main__":
    main()