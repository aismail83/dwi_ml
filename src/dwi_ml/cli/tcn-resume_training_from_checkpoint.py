#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Resume training of a TCN‑GAT model from a checkpoint.
"""
import argparse
import logging
import os

# comet_ml must be imported before torch (see bug report)
import comet_ml

# Hide a harmless PyTorch warning
import warnings
warnings.filterwarnings("ignore",
                        message="`torch.distributed.reduce_op` is deprecated")

from scilpy.io.utils import add_verbose_arg

from dwi_ml.data.dataset.utils import prepare_multisubjectdataset
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.models.projects.tcn_learn2track_model import TCNLearn2TrackModel
from dwi_ml.training.batch_loaders import DWIMLBatchLoaderOneInput
from dwi_ml.training.batch_samplers import DWIMLBatchIDSampler
from dwi_ml.training.projects.tcn_trainer import TCNLearn2TrackTrainer
from dwi_ml.training.utils.experiment import add_args_resuming_experiment
from dwi_ml.training.utils.trainer import run_experiment


def prepare_arg_parser():
    """Build argument parser for resuming training."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    add_args_resuming_experiment(p)   # experiment path, name, patience, etc.
    add_verbose_arg(p)
    return p


def init_from_checkpoint(args, checkpoint_path):
    """
    Reconstruct dataset, model, sampler, loader, and trainer from checkpoint.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments.
    checkpoint_path : str
        Path to the checkpoint directory (usually "checkpoint").

    Returns
    -------
    trainer : TCNGATLearn2TrackTrainer
        Trainer ready to resume training.
    """
    # Load checkpoint state (hyperparameters, epoch, optimizer state, etc.)
    checkpoint_state = TCNLearn2TrackTrainer.load_params_from_checkpoint(
        args.experiments_path, args.experiment_name)

    # Stop if early stopping was triggered previously
    TCNLearn2TrackTrainer.check_stopping_cause(
        checkpoint_state, args.new_patience, args.new_max_epochs)

    # Prepare dataset using the saved parameters
    args_data = checkpoint_state['dataset_params']
    if args.hdf5_file is not None:
        # Override HDF5 file if provided
        args_data['hdf5_file'] = args.hdf5_file
    dataset = prepare_multisubjectdataset(argparse.Namespace(**args_data))

    # Determine logging level for sub‑loggers
    sub_loggers_level = args.verbose if args.verbose != 'DEBUG' else 'INFO'

    # Load TCN‑GAT model from the checkpoint directory
    model = TCNLearn2TrackModel.load_model_from_params_and_state(
        os.path.join(checkpoint_path, 'model'), sub_loggers_level)

    # Rebuild batch sampler from saved parameters
    batch_sampler = DWIMLBatchIDSampler.init_from_checkpoint(
        dataset, checkpoint_state['batch_sampler_params'], sub_loggers_level)

    # Rebuild batch loader from saved parameters
    batch_loader = DWIMLBatchLoaderOneInput.init_from_checkpoint(
        dataset, model, checkpoint_state['batch_loader_params'],
        sub_loggers_level)

    # Instantiate trainer from checkpoint
    with Timer("\nPreparing trainer", newline=True, color='red'):
        trainer = TCNLearn2TrackTrainer.init_from_checkpoint(
            model, args.experiments_path, args.experiment_name,
            batch_sampler, batch_loader,
            checkpoint_state, args.new_patience, args.new_max_epochs,
            args.verbose)

    return trainer


def main():
    """Parse arguments, reconstruct components, and resume training."""
    p = prepare_arg_parser()
    args = p.parse_args()

    # Set root logger to WARNING to avoid too much output from sub‑modules
    logging.getLogger().setLevel(level=logging.WARNING)

    # Locate the checkpoint directory
    checkpoint_path = os.path.join(
        args.experiments_path, args.experiment_name, "checkpoint")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Experiment's checkpoint not found ({})."
                                .format(checkpoint_path))

    trainer = init_from_checkpoint(args, checkpoint_path)

    # Resume training
    run_experiment(trainer)


if __name__ == '__main__':
    main()