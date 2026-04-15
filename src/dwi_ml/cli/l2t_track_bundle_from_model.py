#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script allows tracking from a trained Learn2track model.
"""
import argparse
import logging
import os
from collections import defaultdict
from nibabel.streamlines import Tractogram
import numpy as np
import torch
from dipy.io.stateful_tractogram import (Space, Origin, set_sft_logger_level,
                                         StatefulTractogram)
from dipy.io.streamline import save_tractogram
import dipy.core.geometry as gm
from dipy.io.utils import is_header_compatible
import h5py
import nibabel as nib


from scilpy.io.utils import (add_sphere_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             verify_compression_th)
from scilpy.tracking.utils import (add_seeding_options,
                                   verify_streamline_length_options,
                                   verify_seed_options, add_out_options)

from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.models.projects.learn2track_model import Learn2TrackModel
from dwi_ml.testing.utils import prepare_dataset_one_subj, \
    find_hdf5_associated_to_experiment
from dwi_ml.tracking.projects.learn2track_tracker import RecurrentTracker
from dwi_ml.tracking.tracking_mask import TrackingMask
from dwi_ml.tracking.io_utils import (add_tracking_options,
                                      prepare_seed_generator,
                                      prepare_tracking_mask, track_and_save)

ALWAYS_VOX_SPACE = Space.VOX
ALWAYS_CORNER = Origin('corner')
# Also, after upgrading torch, I now have a lot of warnings:
# FutureWarning: `torch.distributed.reduce_op` is deprecated, please use
# `torch.distributed.ReduceOp` instead
# But I don't use torch.distributed anywhere. Comes from inside torch.
# Hiding warnings for now.
import warnings
warnings.filterwarnings("ignore",
                        message="`torch.distributed.reduce_op` is deprecated")


def build_argparser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    p.add_argument(
            '--out_bundle_dir',
            default=None,
            help='Directory where one .trk per predicted bundle will be saved.'
        )
    track_g = add_tracking_options(p)
    # Sphere used if the direction_getter key is the sphere-classification.
    add_sphere_arg(track_g, symmetric_only=False)

    # As in scilpy:
    add_seeding_options(p)
    add_out_options(p)  # Formatting a bit ugly compared to us, but ok.

    add_verbose_arg(p)

    return p


def prepare_tracker(parser, args):
    hdf5_file = args.hdf5_file or find_hdf5_associated_to_experiment(
        args.experiment_path)
    hdf_handle = h5py.File(hdf5_file, 'r')

    sub_loggers_level = args.verbose if args.verbose != 'DEBUG' else 'INFO'

    with Timer("\nLoading data and preparing tracker...",
               newline=True, color='green'):
        logging.info("Loading seeding mask + preparing seed generator.")
        # Vox space, corner origin
        seed_generator, nbr_seeds, seeding_mask_header, ref = \
            prepare_seed_generator(parser, args, hdf_handle)
        dim = ref.shape

        if args.tracking_mask_group is not None:
            logging.info("Loading tracking mask.")
            tracking_mask, ref2 = prepare_tracking_mask(
                hdf_handle, args.tracking_mask_group, args.subj_id,
                args.mask_interp)

            # Comparing tracking and seeding masks
            is_header_compatible(ref2, seeding_mask_header)
        else:
            tracking_mask = TrackingMask(dim)

        logging.info("Loading subject's data.")
        subset = prepare_dataset_one_subj(
            hdf5_file, args.subj_id, lazy=False,
            cache_size=args.cache_size, subset_name=args.subset,
            volume_groups=[args.input_group], streamline_groups=[])

        logging.info("Loading model.")
        if args.use_latest_epoch:
            model_dir = os.path.join(args.experiment_path, 'best_model')
        else:
            model_dir = os.path.join(args.experiment_path, 'checkpoint/model')
        model = Learn2TrackModel.load_model_from_params_and_state(
            model_dir, log_level=sub_loggers_level)
        logging.info("* Formatted model: " +
                     format_dict_to_str(model.params_for_checkpoint))

        theta = gm.math.radians(args.theta)
        logging.debug("Instantiating tracker.")
        append_last_point = not args.discard_last_point
        tracker = RecurrentTracker(
            input_volume_group=args.input_group,
            dataset=subset, subj_idx=0, model=model, mask=tracking_mask,
            seed_generator=seed_generator, nbr_seeds=nbr_seeds,
            min_len_mm=args.min_length, max_len_mm=args.max_length,
            compression_th=args.compress_th, nbr_processes=args.nbr_processes,
            save_seeds=args.save_seeds, rng_seed=args.rng_seed,
            track_forward_only=args.track_forward_only,
            step_size_mm=args.step_size, algo=args.algo, theta=theta,
            use_gpu=args.use_gpu, eos_stopping_thresh=args.eos_stop,
            simultaneous_tracking=args.simultaneous_tracking,
            append_last_point=append_last_point,
            log_level=args.verbose)

    return tracker, ref


def predict_bundles_for_streamlines(streamlines, tracker, batch_size=64):
    """
    Predict bundle IDs for a list of streamlines using the trained model.

    Parameters
    ----------
    streamlines : list
        List of streamlines (each is an array of shape [N, 3]).
    tracker : object
        Tracker containing the model, dataset and device.
    batch_size : int
        Number of streamlines processed per batch.

    Returns
    -------
    np.ndarray
        Array of predicted bundle IDs (shape: [nb_streamlines]).
    """
    model = tracker.model
    model.eval()
    model.set_context("visu")

    device = tracker.device
    pred_ids = []

    with torch.no_grad():
        for i in range(0, len(streamlines), batch_size):
            # Select batch
            batch_np = streamlines[i:i + batch_size]

            # Convert to tensors
            batch_lines = [
                torch.as_tensor(sl, dtype=torch.float32, device=device)
                for sl in batch_np
            ]

            # Prepare model inputs
            inputs = model.prepare_batch_one_input(
                batch_lines,
                tracker.dataset,
                tracker.subj_idx,
                tracker.volume_group
            )

            # Forward pass
            _, _, bundle_logits = model(
                inputs,
                input_streamlines=batch_lines,
                bundle_ids=None,
                hidden_recurrent_states=True,
                return_hidden=True,
                point_idx=-1
            )

            # Ensure model predicts bundles
            if bundle_logits is None:
                raise ValueError(
                    "bundle_logits is None. Model is not configured for bundle prediction."
                )

            # Get predicted class (argmax)
            batch_pred = torch.argmax(bundle_logits, dim=1)

            # Store results
            pred_ids.extend(batch_pred.cpu().numpy())

    return np.asarray(pred_ids, dtype=np.int64)

def save_streamlines_by_bundle(streamlines, pred_bundle_ids, args, ref, seeds=None):
    """
    Save one tractogram file per predicted bundle.

    Parameters
    ----------
    streamlines : list
        List of generated streamlines.
    pred_bundle_ids : np.ndarray
        Predicted bundle ID for each streamline.
    args : argparse.Namespace
        Script arguments. Must contain out_bundle_dir, hdf5_file,
        experiment_path and save_seeds.
    ref : object
        Reference object used by StatefulTractogram.
    seeds : list, optional
        List of seeds, one per streamline, used only if save_seeds is enabled.
    """
    # Create output directory if it does not exist
    os.makedirs(args.out_bundle_dir, exist_ok=True)

    # Find the HDF5 file used by the experiment
    hdf5_file = args.hdf5_file or find_hdf5_associated_to_experiment(
        args.experiment_path
    )

    # Load bundle names from the HDF5 file
    with h5py.File(hdf5_file, 'r') as hdf_handle:
        grp = hdf_handle["bundle_dict/streamlines"]
        ids = grp["ids"][:]
        names = grp["names"][:]
        names = [
            n.decode("utf-8") if isinstance(n, bytes) else str(n)
            for n in names
        ]
        id_to_name = dict(zip(ids.tolist(), names))

    # Group streamlines by predicted bundle ID
    bundle_streamlines = defaultdict(list)
    bundle_seeds = defaultdict(list)

    for i, (sl, bid) in enumerate(zip(streamlines, pred_bundle_ids)):
        bid = int(bid)
        bundle_streamlines[bid].append(sl)

        if args.save_seeds:
            bundle_seeds[bid].append(
                np.asarray(seeds[i], dtype=np.float32) - 0.5
            )

    # Save one file per bundle
    for bid, sl_list in bundle_streamlines.items():
        bundle_name = id_to_name.get(bid, f"bundle_{bid}")
        filename = f"{bundle_name}.trk"
        out_path = os.path.join(args.out_bundle_dir, filename)

        data_per_streamline = {}
        if args.save_seeds:
            data_per_streamline["seeds"] = bundle_seeds[bid]

        sft = StatefulTractogram(
            sl_list,
            ref,
            space=ALWAYS_VOX_SPACE,
            origin=ALWAYS_CORNER,
            data_per_streamline=data_per_streamline
        )

        save_tractogram(sft, out_path, bbox_valid_check=False)

    print("Finished saving streamlines by bundle.")
    
def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.out_bundle_dir is None:
        parser.error("--out_bundle_dir is required.")

    logging.getLogger().setLevel(level=args.verbose)

    if not nib.streamlines.is_supported(args.out_tractogram):
        parser.error(
            'Invalid output streamline file format (must be trk or tck): '
            f'{args.out_tractogram}'
        )

    assert_inputs_exist(parser, [], args.hdf5_file)
    assert_outputs_exist(parser, args, args.out_tractogram)

    verify_streamline_length_options(parser, args)
    verify_compression_th(args.compress_th)
    verify_seed_options(parser, args)

    tracker, ref = prepare_tracker(parser, args)

    # Tracking
    streamlines, seeds = tracker.track()

    # Classification finale
    pred_bundle_ids = predict_bundles_for_streamlines(streamlines, tracker)

    # Sauvegarde par bundle
    save_streamlines_by_bundle(
        streamlines,
        pred_bundle_ids,
        args,
        ref,
        seeds=seeds if args.save_seeds else None
    )
if __name__ == "__main__":
    main()
