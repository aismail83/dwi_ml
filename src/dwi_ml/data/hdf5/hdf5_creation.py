# -*- coding: utf-8 -*-
import datetime
import glob
import logging
import os
from pathlib import Path
from typing import List

from dipy.io.stateful_tractogram import set_sft_logger_level, Space
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.utils import is_header_compatible
from dipy.tracking.utils import length
import h5py
from scilpy.image.labels import get_data_as_labels

from dwi_ml.data.hdf5.utils import format_nb_blocs_connectivity
from dwi_ml.data.processing.streamlines.data_augmentation import \
    resample_or_compress
from nested_lookup import nested_lookup
import nibabel as nib
import numpy as np

from scilpy.tractograms.tractogram_operations import concatenate_sft

from dwi_ml.data.io import load_file_to4d
from dwi_ml.data.processing.dwi.dwi import standardize_data


def format_filelist(filenames, enforce_presence, folder=None) -> List[str]:
    """
    If folder is not None, it will be added as prefix to all files.
    """
    if isinstance(filenames, str):
        filenames = [filenames]

    new_files = []
    for i, f in enumerate(filenames):
        if folder is not None:
            f = str(folder.joinpath(f))
        if '*' in f:
            tmp = glob.glob(f)
            if len(tmp) == 0:
                msg = "File not found, even with the wildcard: {}".format(f)
                if enforce_presence:
                    raise FileNotFoundError(msg)
                else:
                    logging.warning(msg)
            else:
                new_files.extend(tmp)
        else:
            if not Path(f).is_file():
                msg = "File not found: {}".format(f)
                if enforce_presence:
                    raise FileNotFoundError(msg)
                else:
                    logging.warning(msg)
            else:
                new_files.append(f)
    return new_files


def _load_and_verify_file(filename: str, group_name: str, group_affine,
                          group_res):
    """
    Loads a (3D or 4D) nifti file. If it is a 3D dataset, adds a dimension to
    make it 4D. Then checks that it is compatible with a given group based on
    its affine and resolution.

    Params
    ------
    filename: str
        File's name. Must be .nii or .nii.gz.
    group_name: str
        Name of the group with which 'filename' file must be compatible.
    group_affine: np.array
        The loaded file's affine must be equal (or very close) to this affine.
    group_res: np.array
        The loaded file's resolution must be equal (or very close) to this res.
    """
    if not os.path.isfile(filename):
        logging.debug("      Skipping file {} because it was not "
                      "found in this subject's folder".format(filename))
        # Note: if args.enforce_files_presence was set to true, this
        # case is not possible, already checked in
        # create_hdf5_dataset.py.
        return None

    data, affine, res, _ = load_file_to4d(filename)

    if not np.allclose(affine, group_affine, atol=1e-5):
        # Note. When keeping default options on tolerance, we have run
        # into some errors in some cases, depending on how data has
        # been processed. Now accepting bigger error.
        raise ValueError(
            'Data file {} does not have the same affine as other '
            'files in group {}. Data from each group will be '
            'concatenated, and should have the same affine and voxel '
            'resolution.\n'
            'Affine: {}\n'
            'Group affine: {}\n'
            'Biggest difference: {}'
            .format(filename, group_name, affine, group_affine,
                    np.max(affine - group_affine)))

    if not np.allclose(res, group_res):
        raise ValueError(
            'Data file {} does not have the same resolution as other '
            'files in group {}. Data from each group will be '
            'concatenated, and should have the same affine and voxel '
            'resolution.\n'
            'Resolution: {}\n'
            'Group resolution: {}'
            .format(filename, group_name, res, group_res))

    return data


class HDF5Creator:
    """
    Creates a hdf5 file with:
    - One group per subject:
            - One group per 'volume' group in the config file, where 4D-data is
            the concatenation of every MRI volume listed for this group.
                    -> Volumes will be standardized as defined in the config
                    file (options could be different for each group).
            - One group per 'streamlines' group where data is the decomposed
            SFT containing concatenation of all tractograms listed for this
            group.
                    -> SFTs will be resampled / compressed as defined in the
                    class's arguments (options are the same for all
                    tractograms).

    See the doc for an example of config file.
    https://dwi-ml.readthedocs.io/en/latest/config_file.html
    """
    def __init__(self, root_folder: Path, out_hdf_filename: Path,
                 training_subjs: List[str], validation_subjs: List[str],
                 testing_subjs: List[str], groups_config: dict,
                 step_size: float = None,
                 nb_points: int = None,
                 compress: float = None,
                 remove_invalid: bool = False,
                 enforce_files_presence: bool = True,
                 save_intermediate: bool = False,
                 intermediate_folder: Path = None):
        """
        Params step_size, nb_points and compress are mutually exclusive.

        Params
        ------
        root_folder: Path
            Path to the dwi_ml_ready folder containing all data. See the doc
            for the suggested data organization.
        out_hdf_filename: Path
            Path + filename where to save the final hdf5 file.
        training_subjs: List[str],
        validation_subjs: List[str]
        testing_subj: List[str]
            List of subject names for each data set.
        groups_config: dict
            Information from json file loaded as a dict.
        step_size: float
            Step size to resample streamlines. Default: None.
        nb_points: int
            Number of points per streamline. Default: None.
        compress: float
            Compress streamlines. Default: None.
        remove_invalid: bool
            Remove invalid streamline. Default: False
        enforce_files_presence: bool
            If true, will stop if some files are not available for a subject.
            Default: True.
        save_intermediate: bool
            If true, intermediate files will be saved for debugging purposes.
            Default: False.
        intermediate_folder: Path
            Path where to save the intermediate files.
        """
        # Mandatory
        self.root_folder = root_folder
        self.out_hdf_filename = out_hdf_filename
        self.training_subjs = training_subjs
        self.validation_subjs = validation_subjs
        self.testing_subjs = testing_subjs
        self.groups_config = groups_config
        self.step_size = step_size
        self.nb_points = nb_points
        self.compress = compress
        self.remove_invalid = remove_invalid
        self.bundle_dicts = {}

        # Optional
        self.save_intermediate = save_intermediate
        self.enforce_files_presence = enforce_files_presence
        self.intermediate_folder = intermediate_folder

        # ------- Reading groups config

        self.volume_groups, self.streamline_groups = \
            self._analyse_config_file()

        # -------- Performing checks
        self._check_streamlines_operations()
        # Check that all subjects exist.
        logging.debug("Preparing hdf5 creator for \n"
                      "  training subjs {}, \n"
                      "  validation subjs {},\n"
                      "  testing subjs {}"
                      .format(training_subjs, validation_subjs, testing_subjs))
        self.all_subjs = self._verify_subjects_list()

        # Check that all files exist
        if enforce_files_presence:
            self._check_files_presence()

    def _analyse_config_file(self):
        """
        Reads the groups config json file and finds:
        - List of groups. Their type should be one of 'volume' or 'streamlines'
        - For volume groups: 'standardization' value should be provided and one
          of 'all', 'independent', 'per_file' or 'none'.

        Returns the list of volume groups and streamline groups.
        """
        volume_groups = []
        streamline_groups = []
        for group in self.groups_config.keys():
            if 'type' not in self.groups_config[group]:
                raise KeyError("Group {}'s type was not defined. It should be "
                               "the group type (either 'volume' or "
                               "'streamlines'). See the doc for a "
                               "groups_config.json example.".format(group))
            if 'files' not in self.groups_config[group]:
                raise KeyError(
                    "Group {}'s files were not defined. It should list "
                    "the files to load and concatenate for this group. "
                    "See the doc for a groups_config.json example."
                    .format(group))

            # Volume groups
            if self.groups_config[group]['type'] == 'volume':
                std_choices = ['all', 'independent', 'per_file', 'none']
                if 'standardization' not in self.groups_config[group]:
                    raise KeyError(
                        "Group {}'s 'standardization' was not defined. It "
                        "should be one of {}. See the doc for a "
                        "groups_config.json example."
                        .format(group, std_choices))
                if self.groups_config[group]['standardization'] not in \
                        std_choices:
                    raise KeyError(
                        "Group {}'s 'standardization' should be one of {}, "
                        "but we got {}. See the doc for a groups_config.json "
                        "example."
                        .format(group, std_choices,
                                self.groups_config[group]['standardization']))
                volume_groups.append(group)

            # Streamline groups
            elif self.groups_config[group]['type'] == 'streamlines':
                streamline_groups.append(group)

            else:
                raise ValueError(
                    "Group {}'s type should be one of volume or streamlines "
                    "but got {}"
                    .format(group, self.groups_config[group]['type']))

        logging.info("Volume groups: {}".format(volume_groups))
        logging.info("Streamline groups: {}".format(streamline_groups))
        return volume_groups, streamline_groups

    def _verify_subjects_list(self):
        """
        Raises error if some subjects do not exit in the root folder. Prints
        logging info if some subjects in the root folder were not chosen.
        """
        # Find list of subjects existing inside folder
        possible_subjs = [str(s.name) for s in Path(self.root_folder).iterdir()
                          if s.is_dir()]
        if len(possible_subjs) == 0:
            raise ValueError('No subject found in dwi_ml folder: '
                             '{}'.format(self.root_folder))

        # Check that no subject was added twice. We do not support it.
        if len(np.unique(self.training_subjs)) != len(self.training_subjs):
            raise ValueError("Some training subjects are written twice!")
        if len(np.unique(self.validation_subjs)) != len(self.validation_subjs):
            raise ValueError("Some validation subjects are written twice!")
        if len(np.unique(self.testing_subjs)) != len(self.testing_subjs):
            raise ValueError("Some testing subjects are written twice!")
        all_subjs = self.training_subjs + self.validation_subjs + \
            self.testing_subjs
        unique_subjs = list(set(all_subjs))
        if len(unique_subjs) != len(all_subjs):
            logging.warning(
                "      CAREFUL! Some subjects were added in two different "
                "lists. It it a better practice to have separate datasets for "
                "training/validation/testing.")

        # Checking that chosen subjects exist.
        non_existing_subjs = [s for s in unique_subjs if s not in
                              possible_subjs]
        if len(non_existing_subjs) > 0:
            raise ValueError(
                'Following subjects were chosen for the hdf5 file but their '
                'folders were not found in {}:\n {}'
                .format(self.root_folder, non_existing_subjs))

        # Checking if some existing subjects were not chosen.
        ignored_subj = [s for s in possible_subjs if s not in unique_subjs]
        if len(ignored_subj) > 0:
            logging.warning(
                "    Careful! NOT processing subjects {} from folder because "
                "they were not included in training set, validation set nor "
                "testing set!".format(ignored_subj))
        return unique_subjs

    def _check_files_presence(self):
        """
        Verifying now the list of files. Prevents stopping after a long
        processing time if a file does not exist.

        The list of files to verify for each subject is :
         - the standardization mask
         - all files in the group_config file
        """
        logging.debug("Verifying files presence")

        def flatten_list(a_list):
            new_list = []
            for element in a_list:
                if isinstance(element, list):
                    new_list.extend(flatten_list(element))
                else:
                    new_list.append(element)
            return new_list

        # concatenating files from all groups files:
        config_file_list = [
            nested_lookup('files', self.groups_config),
            nested_lookup('connectivity_matrix', self.groups_config),
            nested_lookup('connectivity_labels', self.groups_config),
            nested_lookup('std_mask', self.groups_config)]
        config_file_list = flatten_list(config_file_list)

        for subj_id in self.all_subjs:
            subj_input_dir = Path(self.root_folder).joinpath(subj_id)

            # Find subject's files from group_config
            _ = format_filelist(config_file_list,
                                self.enforce_files_presence,
                                folder=subj_input_dir)

    def _check_streamlines_operations(self):
        valid = True
        if self.step_size and self.nb_points:
            valid = False
        elif self.step_size and self.compress:
            valid = False
        elif self.nb_points and self.compress:
            valid = False
        if not valid:
            raise ValueError(
                "Only one option can be chosen: either resampling to "
                "step_size, nb_points or compressing, not both.")

    def create_database(self):
        """
        Generates a hdf5 dataset from a group of subjects. Hdf5 dataset will
        contain one group per subject, and for each, groups as defined in the
        config file.

        If wished, all intermediate steps are saved on disk in the hdf5 folder.
        """
        with h5py.File(self.out_hdf_filename, 'w') as hdf_handle:
            # Save configuration
            now = datetime.datetime.now()
            hdf_handle.attrs['data_and_time'] = now.strftime('%d %B %Y %X')
            hdf_handle.attrs['chosen_subjs'] = self.all_subjs
            hdf_handle.attrs['groups_config'] = str(self.groups_config)
            hdf_handle.attrs['training_subjs'] = self.training_subjs
            hdf_handle.attrs['validation_subjs'] = self.validation_subjs
            hdf_handle.attrs['testing_subjs'] = self.testing_subjs
            hdf_handle.attrs['step_size'] = self.step_size if \
                self.step_size is not None else 'Not defined by user'
            hdf_handle.attrs['nb_points'] = self.nb_points if \
                self.nb_points is not None else 'Not defined by user'
            hdf_handle.attrs['compress'] = self.compress if \
                self.compress is not None else 'Not defined by user'

            
            # Add data one subject at the time
            nb_processed = 0
            nb_subjs = len(self.all_subjs)
            logging.debug("Processing {} subjects : {}"
                          .format(nb_subjs, self.all_subjs))
            for subj_id in self.all_subjs:
                nb_processed += 1
                logging.info("*Processing subject {}/{}: {}"
                             .format(nb_processed, nb_subjs, subj_id))
                self._create_one_subj(subj_id, hdf_handle)

            # Write bundle dictionaries (one per streamline group) at the root level
            self._write_bundle_dicts(hdf_handle)

        logging.info("Saved dataset : {}".format(self.out_hdf_filename))

    def _create_one_subj(self, subj_id, hdf_handle):
        """
        Creating one subject's data as a hdf5 group: main attributes +
        volume group(s) + streamline group(s).
        """
        subj_input_dir = self.root_folder.joinpath(subj_id)

        subj_hdf_group = hdf_handle.create_group(subj_id)

        # Add the subj data based on groups in the json config file
        ref = self._create_volume_groups(subj_id, subj_input_dir,
                                         subj_hdf_group)

        self._create_streamline_groups(ref, subj_input_dir, subj_id,
                                       subj_hdf_group)

    def _create_volume_groups(self, subj_id, subj_input_dir, subj_hdf_group):
        """
        Create the hdf5 groups for all volume groups in the config_file for a
        given subject.

        Saves the attrs 'data', 'affine', 'voxres' (voxel resolution) and
        'nb_feature' (the size of last dimension) for each.
        (+ 'type' = 'volume')
        """
        ref_header = None
        for group in self.volume_groups:
            logging.info("    - Processing volume group '{}'...".format(group))

            (group_data, group_affine,
             group_header, group_res) = self._process_one_volume_group(
                group, subj_id, subj_input_dir)
            if ref_header is None:
                ref_header = group_header
            else:
                if not is_header_compatible(ref_header, group_header):
                    raise ValueError("Some volume groups have incompatible "
                                     "headers for subj {}.".format(subj_id))
            logging.debug('      *Done. Now creating dataset from group.')
            hdf_group = subj_hdf_group.create_group(group)
            hdf_group.create_dataset('data', data=group_data)
            logging.debug('      *Done.')

            # Saving data information.
            subj_hdf_group[group].attrs['affine'] = group_affine
            subj_hdf_group[group].attrs['type'] = self.groups_config[group][
                'type']
            subj_hdf_group[group].attrs['voxres'] = group_res

            # Adding the shape info separately to access it without loading
            # the data (useful for lazy data!).
            subj_hdf_group[group].attrs['nb_features'] = group_data.shape[-1]
        return ref_header

    def _process_one_volume_group(self, group: str, subj_id: str,
                                  subj_input_dir: Path):
        """
        Processes each volume group from the json config file for a given
        subject:
        - Loads data from each file of the group and combine them. All datasets
          from a given group must have the same affine, voxel resolution and
          data shape.
          Note. Wildcards will be replaced by the subject id.
        - Standardizes data.

        Parameters
        ----------
        group: str
            Group name.
        subj_id: str
            The subject's id.
        subj_input_dir: Path
            Path where the files from file_list should be found.

        Returns
        -------
        group_data: np.ndarray
            Group data created by concatenating all files, standardized.
        group_affine: np.ndarray
            Affine for the group.
        """
        std_mask = None
        std_option = 'none'
        if 'standardization' in self.groups_config[group]:
            std_option = self.groups_config[group]['standardization']
        if 'std_mask' in self.groups_config[group]:
            if std_option == 'none':
                logging.warning("You provided a std_mask for volume group {}, "
                                "but std_option is 'none'. Skipping.")
            else:
                # Load subject's standardization mask. Can be a list of files.
                std_masks = self.groups_config[group]['std_mask']
                std_masks = format_filelist(std_masks,
                                            self.enforce_files_presence,
                                            folder=subj_input_dir)
                for mask in std_masks:
                    logging.info("       - Loading standardization mask {}"
                                 .format(os.path.basename(mask)))
                    sub_mask_data = nib.load(mask).get_fdata() > 0
                    if std_mask is None:
                        std_mask = sub_mask_data
                    else:
                        std_mask = np.logical_or(sub_mask_data, std_mask)

        # Get the files and add the subject_dir as prefix.
        file_list = self.groups_config[group]['files']
        file_list = format_filelist(file_list, self.enforce_files_presence,
                                    folder=subj_input_dir)

        # First file will define data dimension and affine
        logging.info("       - Processing file {} (first file=reference) "
                     .format(os.path.basename(file_list[0])))
        group_data, group_affine, group_res, group_header = load_file_to4d(
            file_list[0])

        if std_option == 'per_file':
            logging.debug('      *Standardizing sub-data')
            group_data = standardize_data(group_data, std_mask,
                                          independent=False)

        # Other files must fit (data shape, affine, voxel size)
        # It is not a promise that data has been correctly registered, but it
        # is a minimal check.
        if len(file_list) > 1:
            for file_name in file_list[1:]:
                logging.info("       - Processing file {}"
                             .format(os.path.basename(file_name)))
                data = _load_and_verify_file(file_name, group, group_affine,
                                             group_res)

                if std_option == 'per_file':
                    logging.info('          - Standardizing')
                    data = standardize_data(data, std_mask, independent=False)

                # Append file data to hdf group.
                try:
                    group_data = np.append(group_data, data, axis=-1)
                except ImportError:
                    raise ImportError(
                        'Data file {} could not be added to data group {}. '
                        'Wrong dimensions?'.format(file_name, group))

        # Standardize data (per channel) (if not done 'per_file' yet).
        if std_option == 'independent':
            logging.info('       - Standardizing data on each feature.')
            group_data = standardize_data(group_data, std_mask,
                                          independent=True)
        elif std_option == 'all':
            logging.info('       - Standardizing data as a whole.')
            group_data = standardize_data(group_data, std_mask,
                                          independent=False)
        elif std_option not in ['none', 'per_file']:
            raise ValueError("standardization must be one of "
                             "['all', 'independent', 'per_file', 'none']")

        # Save standardized data
        if self.save_intermediate:
            output_fname = self.intermediate_folder.joinpath(
                subj_id + '_' + group + ".nii.gz")
            logging.debug('      *Saving intermediate files into {}.'
                          .format(output_fname))
            standardized_img = nib.Nifti1Image(group_data, group_affine)
            nib.save(standardized_img, str(output_fname))

        return group_data, group_affine, group_header, group_res
    
    

    def _write_bundle_dicts(self, hdf_handle):
        """
        Write one bundle_dict per streamline group at the HDF5 root:

            /bundle_dict/<group_name>/ids
            /bundle_dict/<group_name>/names

        Must be called AFTER at least one subject has been processed, because
        if the config file contains wildcards, we need to actually process a 
        subject to know the files.
        """
        if not hasattr(self, "bundle_dicts") or not self.bundle_dicts:
            raise RuntimeError(
                "bundle_dicts is empty. Make sure subjects were processed "
                "before calling _write_bundle_dicts()."
            )

        bundle_root = hdf_handle.require_group("bundle_dict")
        str_dtype = h5py.string_dtype(encoding="utf-8")

        for group_name in self.streamline_groups:

            if group_name not in self.bundle_dicts:
                raise RuntimeError(
                    f"bundle_dict for group '{group_name}' was never computed."
                )

            bundle_dict = self.bundle_dicts[group_name]
            grp = bundle_root.require_group(group_name)

            # Ensure consistent ordering with enumerate() logic
            ids = np.array(sorted(bundle_dict.keys()), dtype=np.int32)
            names = np.array([bundle_dict[i] for i in ids], dtype=str_dtype)

            # Overwrite safely if rerunning
            for key in ("ids", "names"):
                if key in grp:
                    del grp[key]

            grp.create_dataset("ids", data=ids)
            grp.create_dataset("names", data=names)

    def _create_streamline_groups(self, ref, subj_input_dir, subj_id,
                                  subj_hdf_group):
        """
        Creates one hdf5 group per streamline group in the config file for a
        given subject.

        Saves the attrs 'space', 'affine', 'dimensions', 'voxel_sizes',
        'voxel_order' (i.e. all the SFT's space attributes), 'data', 'offsets',
        'lengths' and 'euclidean_lengths'.
        (+ 'type' = 'streamlines')

        In short, all the nibabel's ArraySequence attributes are saved to
        eventually recreate an SFT from the hdf5 data.
        """
    
        for group in self.streamline_groups:

            # Add the streamlines data
            logging.info('    - Processing tractograms...')

            if ref is None:
                logging.debug(
                    "No group_header! This means no 'volume' group was added "
                    "in the config_file. If all files are .trk, we can use "
                    "ref 'same' but if some files were .tck, we need a ref!"
                    "Hint: Create a volume group 'ref' in the config file.")
            sft, lengths, connectivity_matrix, conn_info, dps_keys,bundle_dict = (
                self._process_one_streamline_group(
                    subj_input_dir, group, subj_id, ref))
            if group not in self.bundle_dicts:
                self.bundle_dicts[group] = bundle_dict
            streamlines_group = subj_hdf_group.create_group(group)
            streamlines_group.attrs['type'] = 'streamlines'

            # The hdf5 can only store numpy arrays (it is actually the
            # reason why it can fetch only precise streamlines from
            # their ID). We need to deconstruct the sft and store all
            # its data separately to allow reconstructing it later.
            (a, d, vs, vo) = sft.space_attributes
            streamlines_group.attrs['space'] = str(sft.space)
            streamlines_group.attrs['origin'] = str(sft.origin)
            streamlines_group.attrs['affine'] = a
            streamlines_group.attrs['dimensions'] = d
            streamlines_group.attrs['voxel_sizes'] = vs
            streamlines_group.attrs['voxel_order'] = vo

            # This streamline's group connectivity info
            if connectivity_matrix is not None:
                streamlines_group.attrs[
                    'connectivity_matrix_type'] = conn_info[0]
                streamlines_group.create_dataset(
                    'connectivity_matrix', data=connectivity_matrix)
                if conn_info[0] == 'from_labels':
                    streamlines_group.create_dataset(
                        'connectivity_label_volume', data=conn_info[1])
                else:
                    streamlines_group.attrs['connectivity_nb_blocs'] = \
                        conn_info[1]

            # DPP not managed yet!
            if len(sft.data_per_point) > 0:
                logging.debug('sft contained data_per_point. Data not kept.')
                logging.debug("    Including dps \"{}\" in the HDF5."
                              .format(dps_keys))

            # # This streamline's group dps info
            dps_group = streamlines_group.require_group("data_per_streamline")

            # --- Always store bundle_ID (generated internally)
            if "bundle_ID" not in sft.data_per_streamline:
                raise RuntimeError(f"bundle_ID missing for subj={subj_id}, group={group}")

            if "bundle_ID" in dps_group:
                del dps_group["bundle_ID"]
            dps_group.create_dataset("bundle_ID", data=sft.data_per_streamline["bundle_ID"])

            # --- Store requested DPS keys from config (must come from disk)
            for dps_key in dps_keys:
                if dps_key == "bundle_ID":
                    continue  # avoid accidental duplication

                if dps_key not in sft.data_per_streamline:
                    raise KeyError(f"Missing data_per_streamline key '{dps_key}' in SFT")

                if dps_key in dps_group:
                    del dps_group[dps_key]

                dps_group.create_dataset(dps_key, data=sft.data_per_streamline[dps_key])


        
        
            # Accessing private Dipy values, but necessary.
            # We need to deconstruct the streamlines into arrays with
            # types recognizable by the hdf5.
            streamlines_group.create_dataset('data',
                                             data=sft.streamlines._data)
            streamlines_group.create_dataset('offsets',
                                             data=sft.streamlines._offsets)
            streamlines_group.create_dataset('lengths',
                                             data=sft.streamlines._lengths)
            streamlines_group.create_dataset('euclidean_lengths', data=lengths)
            
    def _process_one_streamline_group(
            self, subj_dir: Path, group: str, subj_id: str,
            header: nib.Nifti1Header):
        """
        Loads and processes a group of tractograms and merges all streamlines
        together.

        Parameters
        ----------
        subj_dir : Path
            Path to tractograms folder.
        group: str
            group name
        subj_id: str
            The subject's id.
        header : nib.Nifti1Header
            Reference used to load and send the streamlines in voxel space and
            to create final merged SFT. If the file is a .trk, 'same' is used
            instead.

        Returns
        -------
        final_tractogram : StatefulTractogram
            All streamlines in voxel space.
        output_lengths : List[float]
            The Euclidean length of each streamline
        """
        tractograms = self.groups_config[group]['files']
        dps_keys = []
        if 'dps_keys' in self.groups_config[group]:
            dps_keys = self.groups_config[group]['dps_keys']
            if isinstance(dps_keys, str):
                dps_keys = [dps_keys]
        
        # Silencing SFT's logger if our logging is in DEBUG mode, because it
        # typically produces a lot of outputs!
        set_sft_logger_level('WARNING')

        # Initialize
        final_sft = None
        output_lengths = []
        bundle_dict = {}

        tractograms = format_filelist(tractograms, self.enforce_files_presence, folder=subj_dir)
        
        for bundle_id, tractogram_file in enumerate(tractograms):
            # Load without requiring bundle_ID from disk
            sft = self._load_and_process_sft(tractogram_file, header, dps_keys)

            # Map bundle index to bundle name (filename without extension)
            bundle_dict[bundle_id] = Path(tractogram_file).stem
            if sft is not None:
                # Compute euclidean lengths (rasmm space)
                sft.to_space(Space.RASMM)
                output_lengths.extend(length(sft.streamlines))
                # Sending to common space
                sft.to_vox()
                sft.to_corner()

                
                # Generate bundle_ID
                nb_sl = len(sft.streamlines)
                sft.data_per_streamline["bundle_ID"] = np.full(nb_sl, bundle_id, dtype=np.int16)

                # Add processed tractogram to final big tractogram
                if final_sft is None:
                    final_sft = sft
                else:
                    final_sft = concatenate_sft([final_sft, sft], erase_metadata=False)
                
        if self.save_intermediate:
            output_fname = self.intermediate_folder.joinpath(
                subj_id + '_' + group + '.trk')
            logging.debug("      *Saving intermediate streamline group {} "
                        "into {}.".format(group, output_fname))
            # Note. Do not remove the str below. Does not work well
            # with Path.
            save_tractogram(final_sft, str(output_fname))

        conn_matrix = None
        conn_info = None
        if 'connectivity_matrix' in self.groups_config[group]:
            logging.info("         Now preparing connectivity matrix")
            if not ("connectivity_nb_blocs" in self.groups_config[group] or
                    "connectivity_labels" in self.groups_config[group]):
                raise ValueError(
                    "The config file must provide either the "
                    "connectivity_nb_blocs or the connectivity_labels option "
                    "associated with the streamline group '{}'"
                    .format(group))
            elif ("connectivity_nb_blocs" in self.groups_config[group] and
                    "connectivity_labels" in self.groups_config[group]):
                raise ValueError(
                    "The config file must only provide ONE of the "
                    "connectivity_nb_blocs or the connectivity_labels option "
                    "associated with the streamline group '{}'"
                    .format(group))
            elif "connectivity_nb_blocs" in self.groups_config[group]:
                nb_blocs = format_nb_blocs_connectivity(
                    self.groups_config[group]['connectivity_nb_blocs'])
                conn_info = ['from_blocs', nb_blocs]
            else:  # labels
                labels_file = self.groups_config[group]['connectivity_labels']
                labels_file = os.path.join(subj_dir, labels_file)
                labels_data = get_data_as_labels(nib.load(labels_file))
                conn_info = ['from_labels', labels_data]

            conn_file = subj_dir.joinpath(
                self.groups_config[group]['connectivity_matrix'])
            conn_matrix = np.load(conn_file)
            conn_matrix = conn_matrix > 0

        return final_sft, output_lengths, conn_matrix, conn_info, dps_keys,bundle_dict

    def _load_and_process_sft(self, tractogram_file, header, dps_keys):
        # Check file extension
        _, file_extension = os.path.splitext(str(tractogram_file))
        if file_extension not in ['.trk', '.tck']:
            raise ValueError(
                "We do not support file's type: {}. We only support .trk "
                "and .tck files.".format(tractogram_file))
        if file_extension == '.trk':
            if header and not is_header_compatible(str(tractogram_file),
                                                   header):
                raise ValueError("Streamlines group is not compatible "
                                 "with volume groups\n ({})"
                                 .format(tractogram_file))
            # overriding given header.
            header = 'same'

        # Loading tractogram and sending to wanted space
        logging.info("       - Processing tractogram {}"
                     .format(os.path.basename(tractogram_file)))# -*- coding: utf-8 -*-
"""
Two-block TCN model for tractography.
"""

import logging
from typing import List, Optional, Union
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_sequence
from torch.nn.utils.rnn import invert_permutation
from dwi_ml.data.processing.space.neighborhood import unflatten_neighborhood
from dwi_ml.data.processing.streamlines.post_processing import (
    compute_directions, normalize_directions, compute_n_previous_dirs
)
from dwi_ml.models.main_models import (
    ModelWithPreviousDirections,
    ModelWithDirectionGetter,
    ModelWithNeighborhood,
    ModelWithOneInput
)
from dwi_ml.models.projects.learn2track_model import faster_unpack_sequence

logger = logging.getLogger("model_logger")


class Chomp1d(nn.Module):
    """Removes the extra right padding added to preserve causal convolutions."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class CausalConvBlock(nn.Module):
    """
    A single causal dilated Conv1d block:
    Conv1d -> Chomp -> ReLU -> Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        self.chomp = Chomp1d(padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.chomp(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class TCNSubBlock(nn.Module):
    """
    A TCN sub-block composed of 5 causal dilated convolutions.
    The default dilation factors are [1, 3, 6, 12, 24].
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        kernel_size: int = 6,
        dilations=(1, 3, 6, 12, 24),
        dropout: float = 0.0
    ):
        super().__init__()

        layers = []
        current_in = in_channels
        for d in dilations:
            layers.append(
                CausalConvBlock(
                    in_channels=current_in,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout
                )
            )
            current_in = hidden_channels

        self.network = nn.Sequential(*layers)
        self.output_size = hidden_channels

        self.residual_proj = (
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
            if in_channels != hidden_channels else nn.Identity()
        )

        self.out_relu = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, T).
        mask : Optional[torch.Tensor]
            Optional boolean mask of shape (B, T).
        """
        if mask is not None:
            x = x * mask[:, None, :].float()

        residual = self.residual_proj(x)
        out = self.network(x)
        out = out + residual
        out = self.out_relu(out)

        if mask is not None:
            out = out * mask[:, None, :].float()

        return out


class TCNLearn2TrackModel(
    ModelWithPreviousDirections,
    ModelWithDirectionGetter,
    ModelWithNeighborhood,
    ModelWithOneInput
):
    """
    Two-block TCN model compatible with a Learn2Track-like API.
    """

    def __init__(self,
                 experiment_name,
                 step_size: Union[float, None],
                 compress_lines: Union[float, None],
                 nb_features: int,

                 # PREVIOUS DIRECTIONS
                 nb_previous_dirs: Union[int, None],
                 prev_dirs_embedded_size: Union[int, None],
                 prev_dirs_embedding_key: Union[str, None],
                 normalize_prev_dirs: bool,

                 # INPUT EMBEDDING
                 input_embedding_key: str,
                 input_embedded_size: Union[int, None],
                 nb_cnn_filters: Optional[List[int]],
                 kernel_size: Optional[Union[int, List[int]]],

                 # TCN
                 tcn_hidden_size: int,
                 tcn_num_layers: int,
                 tcn_kernel_size: int,
                 # DIRECTION GETTER
                 dg_key: str,
                 dg_args: Union[dict, None],

                 # NEIGHBORHOOD
                 neighborhood_type: Optional[str] = None,
                 neighborhood_radius: Optional[int] = None,
                 neighborhood_resolution: Optional[float] = None,
                 log_level=logging.root.level,
                 nb_points: Optional[int] = None,

                 # BUNDLE OPTIONS
                 use_bundle_ids: bool = False,
                 bundle_emb_dim: Optional[int] = None,
                 num_bundles: Optional[int] = None,
                 predict_bundle_ids: bool = True,

                 # REGULARIZATION
                 dropout: float = 0.1):

        super().__init__(
            experiment_name=experiment_name,
            step_size=step_size,
            nb_points=nb_points,
            compress_lines=compress_lines,
            log_level=log_level,

            neighborhood_type=neighborhood_type,
            neighborhood_radius=neighborhood_radius,
            neighborhood_resolution=neighborhood_resolution,

            nb_features=nb_features,
            input_embedding_key=input_embedding_key,
            input_embedded_size=input_embedded_size,
            nb_cnn_filters=nb_cnn_filters,
            kernel_size=kernel_size,

            nb_previous_dirs=nb_previous_dirs,
            prev_dirs_embedded_size=prev_dirs_embedded_size,
            prev_dirs_embedding_key=prev_dirs_embedding_key,
            normalize_prev_dirs=normalize_prev_dirs,

            dg_args=dg_args,
            dg_key=dg_key
        )

        if dropout < 0 or dropout > 1:
            raise ValueError("The dropout rate must be between 0 and 1.")

        self.dropout = dropout
        self.log_level = log_level
        self.dg_args = dg_args

        self.use_bundle_ids = bool(use_bundle_ids)
        self.predict_bundle_ids = bool(predict_bundle_ids)

        self.tcn_hidden_size = tcn_hidden_size
        self.tcn_num_layers = tcn_num_layers
        self.tcn_kernel_size = tcn_kernel_size

        if self.predict_bundle_ids and (num_bundles is None or num_bundles <= 0):
            raise ValueError(
                "num_bundles must be provided and > 0 when predict_bundle_ids=True"
            )

        if self.use_bundle_ids:
            if bundle_emb_dim is None:
                raise ValueError(
                    "bundle_emb_dim must be provided when use_bundle_ids=True"
                )
            if num_bundles is None:
                raise ValueError(
                    "num_bundles must be provided when use_bundle_ids=True"
                )

            self.bundle_emb_dim = int(bundle_emb_dim)
            self.num_bundles = int(num_bundles)

            if self.bundle_emb_dim <= 0:
                raise ValueError(
                    f"bundle_emb_dim must be > 0 (got {self.bundle_emb_dim})"
                )
            if self.num_bundles <= 0:
                raise ValueError(
                    f"num_bundles must be > 0 (got {self.num_bundles})"
                )

            self.bundle_emb = nn.Embedding(
                self.num_bundles, self.bundle_emb_dim
            )
        else:
            self.bundle_emb_dim = 0
            self.num_bundles = int(num_bundles) if num_bundles is not None else 0
            self.bundle_emb = None

        self.embedding_dropout = nn.Dropout(self.dropout)

        # Raw input size before the optional embedding stage.
        self.raw_input_size = nb_features * self.nb_neighbors

        # Embedded input size used as the point-wise feature x^(1).
        self.input_size = self.computed_input_embedded_size
        if self.use_bundle_ids:
            self.input_size += self.bundle_emb_dim
        if self.nb_previous_dirs > 0:
            self.input_size += self.prev_dirs_embedded_size

        self.dilations = (1, 3, 6, 12, 24, 48)

        # -------------------------
        # TCN block
        # -------------------------
        self.tcn = TCNSubBlock(
            in_channels=self.input_size,
            hidden_channels=tcn_hidden_size,
            kernel_size=tcn_kernel_size,
            dilations=self.dilations,
            dropout=dropout
        )

        self.context_len = 1 + (tcn_kernel_size) * len(self.dilations)

        # The direction getter takes the TCN block output as input.
        self.instantiate_direction_getter(self.tcn.output_size)

    def set_context(self, context):
        assert context in ['training', 'validation', 'tracking', 'visu',
                           'preparing_backward']
        self._context = context

    @property
    def params_for_checkpoint(self):
        params = super().params_for_checkpoint
        params.update({
            'nb_features': int(self.nb_features),
            'tcn_hidden_size': self.tcn_hidden_size,
            'tcn_num_layers': self.tcn_num_layers,
            'tcn_kernel_size': self.tcn_kernel_size,
            'dropout': self.dropout,
            'use_bundle_ids': self.use_bundle_ids,
            'bundle_emb_dim': self.bundle_emb_dim,
            'num_bundles': self.num_bundles,
            'predict_bundle_ids': self.predict_bundle_ids,
        })
        return params

    @property
    def computed_params_for_display(self):
        p = super().computed_params_for_display
        p['tcn_output_size'] = self.tcn.output_size
        return p

    def _flatten_time_major(self, seq_tensor, batch_sizes):
        """
        Reconstruct PackedSequence.data layout from a padded tensor.

        Parameters
        ----------
        seq_tensor : torch.Tensor
            Tensor of shape (B, T, F).
        batch_sizes : torch.Tensor
            Batch sizes from the packed sequence.

        Returns
        -------
        flat_out : torch.Tensor
            Flattened tensor of shape (sum(lengths), F).
        """
        return torch.cat(
            [seq_tensor[:batch_sizes[t], t, :] for t in range(len(batch_sizes))],
            dim=0
        )

    def forward(self,
            x: List[torch.Tensor],
            input_streamlines: List[torch.Tensor] = None,
            bundle_ids: torch.Tensor = None,
            hidden_recurrent_states: List = None,
            return_hidden: bool = False,
            point_idx: int = None):

        del hidden_recurrent_states
        dev = next(self.parameters()).device

        if self.context is None:
            raise ValueError("Please set context before usage.")

        assert x[0].shape[-1] == self.raw_input_size, \
            "Not the expected input size! Should be {} but got {}.".format(
                self.raw_input_size, x[0].shape[-1])

        unsorted_indices = None
        sorted_indices = None

        # In training and validation, sequences are sorted by length.
        if self.context != 'tracking':
            sort_lengths = torch.as_tensor([len(s) for s in x])
            _, sorted_indices = torch.sort(sort_lengths, descending=True)
            unsorted_indices = invert_permutation(sorted_indices)
            x = [x[i] for i in sorted_indices]
            if input_streamlines is not None:
                input_streamlines = [input_streamlines[i] for i in sorted_indices]

        # -----------------------------------
        # Previous directions: always use the full sequence for the TCN.
        # -----------------------------------
        n_prev_dirs = None
        if self.nb_previous_dirs > 0:
            if input_streamlines is None:
                raise ValueError(
                    "input_streamlines must be provided when nb_previous_dirs > 0"
                )

            dirs = compute_directions(input_streamlines)
            if self.normalize_prev_dirs:
                dirs = normalize_directions(dirs)

            # Important: preserve full temporal alignment for the TCN.
            n_prev_dirs = compute_n_previous_dirs(
                dirs, self.nb_previous_dirs, point_idx=None
            )
            n_prev_dirs = pack_sequence(n_prev_dirs, enforce_sorted=False)
            n_prev_dirs = self.prev_dirs_embedding(n_prev_dirs.data)
            n_prev_dirs = self.embedding_dropout(n_prev_dirs)

        # -----------------------------------
        # Pack the input sequences
        # -----------------------------------
        x_packed = pack_sequence(x)
        batch_sizes = x_packed.batch_sizes
        x_data = x_packed.data

        # -----------------------------------
        # Compute input embeddings
        # -----------------------------------
        if self.input_embedding_key == 'cnn_embedding':
            x_data = unflatten_neighborhood(
                x_data, self.neighborhood_vectors,
                self.neighborhood_type, self.neighborhood_radius,
                self.neighborhood_resolution
            )

        x_data = self.input_embedding_layer(x_data)
        x_data = self.embedding_dropout(x_data)

        if n_prev_dirs is not None:
            assert x_data.shape[0] == n_prev_dirs.shape[0], \
                f"x_data: {x_data.shape}, n_prev_dirs: {n_prev_dirs.shape}"
            x_data = torch.cat((x_data, n_prev_dirs), dim=-1)

        expected_size = self.input_size
        got_size = x_data.shape[-1]
        if got_size != expected_size:
            raise ValueError(
                f"Wrong feature size before Block 1: expected "
                f"{expected_size}, got {got_size}."
            )

        # Reconstruct the per-sequence list after embedding.
        seq_features = faster_unpack_sequence(
            PackedSequence(
                x_data,
                batch_sizes,
                x_packed.sorted_indices,
                x_packed.unsorted_indices
            )
        )

        # -----------------------------------
        # Pad the sequences
        # -----------------------------------
        padded_x1 = pad_sequence(seq_features, batch_first=True)  # (B, T, F1)
        batch_size, max_len, _ = padded_x1.shape

        mask = torch.zeros(
            batch_size, max_len, dtype=torch.bool, device=dev
        )
        seq_lengths = []
        for i, seq in enumerate(seq_features):
            cur_len = len(seq)
            seq_lengths.append(cur_len)
            mask[i, :cur_len] = True

        # -----------------------------------
        # TCN block
        # -----------------------------------
        tcn_in = padded_x1.permute(0, 2, 1).contiguous()   # (B, F, T)
        tcn_out = self.tcn(tcn_in)                         # (B, H, T)
        tcn_out = tcn_out.permute(0, 2, 1).contiguous()   # (B, T, H)

        # -----------------------------------
        # Prepare the input for the direction getter
        # -----------------------------------
        if point_idx is None:
            dg_in = self._flatten_time_major(tcn_out, batch_sizes)
        else:
            lengths_t = torch.as_tensor(seq_lengths, device=dev, dtype=torch.long)

            if point_idx < 0:
                gather_idx = lengths_t + point_idx   # e.g. -1 => last valid point
            else:
                gather_idx = torch.full(
                    (batch_size,), point_idx, device=dev, dtype=torch.long
                )

            gather_idx = torch.clamp(gather_idx, min=0)
            gather_idx = torch.minimum(gather_idx, lengths_t - 1)

            dg_in = tcn_out[
                torch.arange(batch_size, device=dev),
                gather_idx,
                :
            ]   # (B, H)

        assert dg_in.shape[-1] == self.direction_getter.input_size, \
            "Expecting input to direction getter of size {}. Got {}.".format(
                self.direction_getter.input_size, dg_in.shape[-1]
            )

        model_outputs = self.direction_getter(dg_in)

        # -----------------------------------
        # In tracking mode, return the raw tensor
        # (one output per active streamline).
        # -----------------------------------
        if self.context == 'tracking':
            return model_outputs, None, None

        # -----------------------------------
        # Outside tracking mode, restore the original structure.
        # -----------------------------------
        if point_idx is not None:
            if 'gaussian' in self.dg_key or 'fisher' in self.dg_key:
                x1, x2 = model_outputs
                model_outputs = (
                    [x1[i].unsqueeze(0) for i in unsorted_indices],
                    [x2[i].unsqueeze(0) for i in unsorted_indices]
                )
            else:
                model_outputs = [
                    model_outputs[i].unsqueeze(0) for i in unsorted_indices
                ]
        else:
            if 'gaussian' in self.dg_key or 'fisher' in self.dg_key:
                x1, x2 = model_outputs

                x1 = PackedSequence(
                    x1, batch_sizes,
                    x_packed.sorted_indices, x_packed.unsorted_indices
                )
                x2 = PackedSequence(
                    x2, batch_sizes,
                    x_packed.sorted_indices, x_packed.unsorted_indices
                )

                x1 = faster_unpack_sequence(x1)
                x2 = faster_unpack_sequence(x2)

                model_outputs = (
                    [x1[i] for i in unsorted_indices],
                    [x2[i] for i in unsorted_indices]
                )
            else:
                model_outputs = PackedSequence(
                    model_outputs,
                    batch_sizes,
                    x_packed.sorted_indices,
                    x_packed.unsorted_indices
                )
                model_outputs = faster_unpack_sequence(model_outputs)
                model_outputs = [model_outputs[i] for i in unsorted_indices]

        return model_outputs, None, None
        sft = load_tractogram(str(tractogram_file), header,bbox_valid_check=False)

        # Check for required dps_keys
        for dps_key in dps_keys:
            if dps_key not in sft.data_per_streamline.keys():
                raise ValueError("DPS key {} is not present in file {}. Only "
                                 "found the following keys: {}"
                                 .format(dps_key, tractogram_file,
                                         list(sft.data_per_streamline.keys())))

        # Remove non-required dps_keys
        for dps_key in list(sft.data_per_streamline.keys()):
            if dps_key not in dps_keys:
                del sft.data_per_streamline[dps_key]

        # Resample or compress streamlines
        sft = resample_or_compress(sft, self.step_size,
                                   self.nb_points,
                                   self.compress,
                                   self.remove_invalid)

        return sft