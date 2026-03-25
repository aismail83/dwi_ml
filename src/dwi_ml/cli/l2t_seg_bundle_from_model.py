#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script allows tracking from a trained Learn2track model.
"""
import argparse
import logging
from tqdm import tqdm
import os
from collections import defaultdict
from nibabel.streamlines import Tractogram
import numpy as np
import torch
from dipy.io.stateful_tractogram import (Space, Origin, set_sft_logger_level, StatefulTractogram)

from dipy.io.streamline import save_tractogram,load_tractogram
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


def prepare_model(parser, args):
    hdf5_file = args.hdf5_file or find_hdf5_associated_to_experiment(
        args.experiment_path)
    hdf_handle = h5py.File(hdf5_file, 'r')

    sub_loggers_level = args.verbose if args.verbose != 'DEBUG' else 'INFO'

    with Timer("\nLoading data and preparing model...",
               newline=True, color='green'):

        logging.info("Loading seeding mask + preparing seed generator.")
        # Vox space, corner origin
        seed_generator, nbr_seeds, seeding_mask_header, ref = \
            prepare_seed_generator(parser, args, hdf_handle)
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

        

    return model, subset,ref


def predict_bundles_for_streamlines(
    streamlines,
    model,
    dataset,
    subj_idx,
    volume_group_idx,
    batch_size=64
    ):
    """
    Predict bundle IDs for a list of streamlines using the trained model.

    Parameters
    ----------
    streamlines : list
        List of streamlines (each streamline has shape [N, 3]).
    model : Learn2TrackModel
        Trained model.
    dataset : object
        Dataset used to prepare the model inputs.
    subj_idx : int
        Subject index in the dataset.
    volume_group_idx : int
        Volume group index in the dataset.
    batch_size : int
        Number of streamlines processed per batch.

    Returns
    -------
    np.ndarray
        Predicted bundle IDs, shape [nb_streamlines].
    """
    model.eval()
    model.set_context("tracking")

    device = next(model.parameters()).device
    pred_ids = []

    for i in tqdm(
        range(0, len(streamlines), batch_size),
        desc="Predicting bundles",
        unit="batch"
    ):
        batch_np = streamlines[i:i + batch_size]

        batch_lines = [
            torch.as_tensor(sl, dtype=torch.float32, device=device)
            for sl in batch_np
        ]

        with torch.no_grad():
            inputs = model.prepare_batch_one_input(
                batch_lines,
                dataset,
                subj_idx,
                volume_group_idx
            )
            
            _, _, bundle_logits = model(
                inputs,
                input_streamlines=batch_lines,
                hidden_recurrent_states=None,
                return_hidden=False,
                point_idx=None
            )

        if bundle_logits is None:
            raise ValueError(
                "bundle_logits is None. Model is not configured for bundle prediction."
            )

        batch_pred = torch.argmax(bundle_logits, dim=1)
        pred_ids.extend(batch_pred.detach().cpu().numpy().tolist())

    return np.asarray(pred_ids, dtype=np.int64)


def save_streamlines_by_bundle(streamlines, pred_bundle_ids, ref, args, seeds=None):
    """
    Save one tractogram file per predicted bundle.

    Parameters
    ----------
    streamlines : list
        List of generated streamlines.
    pred_bundle_ids : np.ndarray
        Predicted bundle ID for each streamline.
    ref : object
        Reference object used by StatefulTractogram.
    args : argparse.Namespace
        Script arguments.
    seeds : list, optional
        List of seeds, one per streamline, used only if save_seeds is enabled.
    """
    os.makedirs(args.out_bundle_dir, exist_ok=True)

    hdf5_file = args.hdf5_file or find_hdf5_associated_to_experiment(
        args.experiment_path
    )

    with h5py.File(hdf5_file, 'r') as hdf_handle:
        grp = hdf_handle["bundle_dict/streamlines"]
        ids = grp["ids"][:]
        names = grp["names"][:]
        names = [
            n.decode("utf-8") if isinstance(n, bytes) else str(n)
            for n in names
        ]
        id_to_name = dict(zip(ids.tolist(), names))

    if len(streamlines) != len(pred_bundle_ids):
        raise ValueError(
            f"Mismatch: {len(streamlines)} streamlines but "
            f"{len(pred_bundle_ids)} predicted bundle IDs."
        )

    bundle_streamlines = defaultdict(list)
    bundle_seeds = defaultdict(list)

    for i, (sl, bid) in enumerate(zip(streamlines, pred_bundle_ids)):
        bid = int(bid)
        bundle_streamlines[bid].append(sl)

        if args.save_seeds:
            if seeds is None:
                raise ValueError("args.save_seeds=True but seeds=None.")
            bundle_seeds[bid].append(
                np.asarray(seeds[i], dtype=np.float32) - 0.5
            )

    bundle_items = list(bundle_streamlines.items())

    for bid, sl_list in tqdm(bundle_items, desc="Saving bundles", unit="bundle"):
        bundle_name = id_to_name.get(bid, f"bundle_{bid}")
        filename = f"{bundle_name}.trk"
        out_path = os.path.join(args.out_bundle_dir, filename)

        data_per_streamline = {}
        if args.save_seeds:
            data_per_streamline["seeds"] = bundle_seeds[bid]

        sft = StatefulTractogram(
            streamlines=sl_list,
            reference=ref,
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

    

    model, dataset,ref= prepare_model(parser, args)

    # Load existing tractogram
    sft = load_tractogram(args.out_tractogram, 'same')
    sft.to_vox()
    sft.to_corner()
    streamlines = list(sft.streamlines)
    subj_idx = 0
    volume_group_idx = dataset.volume_groups.index(args.input_group)
    print("Loaded tractogram:", len(streamlines), "streamlines")
   
    # Predict bundles
    logging.debug("Instantiating pred_bundle_ids.")
    pred_bundle_ids = predict_bundles_for_streamlines(streamlines, model,dataset, subj_idx,volume_group_idx
    , batch_size=256)
    
    # Save bundles
    save_streamlines_by_bundle(streamlines, pred_bundle_ids, ref, args )
if __name__ == "__main__":
    main()
