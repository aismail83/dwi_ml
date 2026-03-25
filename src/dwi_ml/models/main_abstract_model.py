# -*- coding: utf-8 -*-
import json
import logging
import os
import shutil
from argparse import ArgumentParser
import torch.nn.functional as F
import torch
import h5py


from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.io_utils import add_resample_or_compress_arg
from dwi_ml.data.dataset.multi_subject_containers import MultisubjectSubset
logger = logging.getLogger('model_logger')


class MainModelAbstract(torch.nn.Module):
    """
    Parent model that should be used for all models, in all projects.
    - Defines the way to save the model at checkpoints and after training with
      methods:
        - save_params_and_state
        - load_model_from_params_and_state
    - Prepares a method 'compute_loss', which will be called by our trainer
      during training. Should be defined by all child classes
    - Defines the way to get streamlines from the BatchLoader, with parameters
      such as step_size, nb_points or compress_lines. The preprocessing steps
      are performed by the BatchLoader, but it probably influences strongly
      how the model performs, particularly in sequence-based models, as it
      changes the length of streamlines. When using a fully trained model in
      various scripts, you will often have the option to modify this value in
      the BatchLoader, but it is probably not recommanded. This is why it has
      been added as a main hyperparameter.
    - Defines the type of inputs the forward() method will receive when called
      in the trainer.
    - Adds some internal values for easier management, such as self.device and
      self.context.
    """
    subset: MultisubjectSubset
    def __init__(self, experiment_name: str,
                 # Target preprocessing params for the BatchLoader + tracker
                subset: MultisubjectSubset=None,
                 step_size: float = None,
                
                 nb_points: int = None,
                 compress_lines: float = False,
                 # Other
                 log_level=logging.root.level):
        """
        Params
        ------
        experiment_name: str
            Name of the experiment
        step_size : float
            Constant step size that every streamline should have between points
            (in mm). Default: None.
        nb_points: int
            Alternatively, mandatory number of points per streamline.
        compress_streamlines: float
            If set, compress streamlines to that tolerance error. Cannot be
            used together with step_size. This model cannot be used for
            tracking.
        log_level: str
            Level of the model logger. Default: root's level.
        """
        super().__init__()
        
        self.subset= subset
        self.experiment_name = experiment_name
        self.bundle_class_weights = None

        
        # Trainer's logging level can be changed separately from main
        # scripts.
        logger.setLevel(log_level)

        self.device = None

        # To tell our BatchLoader how to resample streamlines during training
        # (should also be the step size during tractography).
        if ((step_size and compress_lines) or (step_size and nb_points) or
                (nb_points and compress_lines)):
            raise ValueError(
                "You may choose either resampling (step_size or nb_points)"
                " or compressing, but not two of them or more.")
        elif step_size and step_size <= 0:
            raise ValueError("Step size can't be 0 or less!")
        elif nb_points and nb_points <= 0:
            raise ValueError("Number of points can't be 0 or less!")
            # Note. When using
            # scilpy.tracking.tools.resample_streamlines_step_size, a warning
            # is shown if step_size < 0.1 or > np.max(sft.voxel_sizes), saying
            # that the value is suspicious. Not raising the same warnings here
            # as you may be wanting to test weird things to understand better
            # your model.
        self.nb_points = nb_points
        self.step_size = step_size
        self.compress_lines = compress_lines

        # Adding a context. Most models in here act differently
        # during training (ex: no loss at the last coordinate = we skip it)
        # vs during tracking (only the last coordinate is important) vs during
        # visualisation (the whole streamline is important).
        self._context = None

    @staticmethod
    def add_args_main_model(p: ArgumentParser):
        """Parameters to add to your scripts for this model, with argparse."""
        add_resample_or_compress_arg(p)

    def set_context(self, context):
        """Sets the context management for models in dwi_ml. Some models may
        deal differently with the data during training or validation. Our
        trainer tells our model the context at the beginning of training and
        at the beginning of validation."""
        assert context in ['training',  'validation']
        self._context = context

    @property
    def context(self):
        "Get the context."
        return self._context

    def move_to(self, device):
        """
        Careful. Calling model.to(a_device) does not influence the self.device.
        Prefer this method for easier management.
        """
        self.to(device, non_blocking=True)
        self.device = device

    @property
    def params_for_checkpoint(self):
        """All parameters necessary to create again the same model. Will be
        used in the trainer, when saving the checkpoint state. Params here
        will be used to re-create the model when starting an experiment from
        checkpoint. You should be able to re-create an instance of your
        model with those params."""
        return {
            'experiment_name': self.experiment_name,
            'step_size': self.step_size,
            'compress_lines': self.compress_lines,
            'nb_points': self.nb_points,
        }

    @property
    def computed_params_for_display(self):
        p = {}
        return p

    def save_params_and_state(self, model_dir):
        model_state = self.state_dict()

        # If a model was already saved, back it up and erase it after saving
        # the new.
        to_remove = None
        if os.path.exists(model_dir):
            to_remove = os.path.join(model_dir, "..", "model_old")
            shutil.move(model_dir, to_remove)
        os.makedirs(model_dir)

        # Save attributes
        name = os.path.join(model_dir, "parameters.json")
        with open(name, 'w') as json_file:
            json_file.write(json.dumps(self.params_for_checkpoint, indent=4,
                                       separators=(',', ': ')))

        name = os.path.join(model_dir, "model_type.txt")
        with open(name, 'w') as txt_file:
            txt_file.write(str(self.__class__.__name__))

        # Save model
        torch.save(model_state, os.path.join(model_dir, "model_state.pkl"))

        if to_remove:
            shutil.rmtree(to_remove)

    @classmethod
    def load_model_from_params_and_state(cls, model_dir,
                                         log_level=logging.WARNING):
        """
        Params
        -----
        loading_dir: path
            Path to the trained parameters, either from the latest checkpoint
            or from the best model folder. Must contain files
            - parameters.json
            - model_state.pkl
        """
        params = cls._load_params(model_dir)

        logger.setLevel(log_level)
        logger.debug("Loading model from saved parameters:" +
                     format_dict_to_str(params))
        params.update(log_level=log_level)
        model = cls(**params)

        model_state = cls._load_state(model_dir)
        model.load_state_dict(model_state)  # using torch's method

        # By default, setting to eval state. If this will be used by the
        # trainer, it will call model.train().
        model.eval()

        return model

    @classmethod
    def _load_params(cls, model_dir):
        # Load attributes and hyperparameters from json file
        params_filename = os.path.join(model_dir, "parameters.json")
        with open(params_filename, 'r') as json_file:
            params = json.load(json_file)

        return params

    @classmethod
    def _load_state(cls, model_dir):
        model_state_file = os.path.join(model_dir, "model_state.pkl")
        model_state = torch.load(model_state_file)

        return model_state

    def forward(self, inputs, streamlines):
        raise NotImplementedError

    def compute_loss(self, model_outputs, target_streamlines):
        raise NotImplementedError
    



    def compute_bundles_class_weights(self, subset, num_classes=21):
        """
        Compute dataset-level class weights from all bundle IDs in the subset.
        """
        if subset is None:
            raise ValueError("subset is None. Pass dataset.training_set.")
        all_train_bundle_ids = []

        with h5py.File(subset.hdf5_file, "r") as f:
            for subj_id in f.keys():  # 🔥 plus sûr que subset.subjects
                subj_group = f[subj_id]

                for group_name in subset.streamline_groups:
                    if group_name not in subj_group:
                        continue

                    streamlines_group = subj_group[group_name]

                    if "data_per_streamline" not in streamlines_group:
                        continue
                    if "bundle_ID" not in streamlines_group["data_per_streamline"]:
                        continue

                    bundle_ids = streamlines_group["data_per_streamline"]["bundle_ID"][:]
                    bundle_ids = torch.as_tensor(bundle_ids, dtype=torch.long).view(-1)
                    all_train_bundle_ids.append(bundle_ids)

        if len(all_train_bundle_ids) == 0:
            raise RuntimeError("No bundle_ID found in the subset.")

        all_train_bundle_ids = torch.cat(all_train_bundle_ids, dim=0)

        class_counts = torch.bincount(all_train_bundle_ids, minlength=num_classes).float()

        class_weights = torch.zeros_like(class_counts)
        nonzero = class_counts > 0
        class_weights[nonzero] = 1.0 / class_counts[nonzero]

        if nonzero.any():
            class_weights[nonzero] = (
                class_weights[nonzero] / class_weights[nonzero].sum() * nonzero.sum()
            )

        return class_weights
    


    def compute_bundle_loss(self, bundle_logits=None, bundle_ids=None) -> torch.Tensor:
        """
        Compute the bundle classification loss.
        """
        if bundle_logits is None or bundle_ids is None:
            if bundle_logits is not None:
                return bundle_logits.new_zeros(())
            device = self.device if hasattr(self, "device") else "cpu"
            return torch.tensor(0.0, device=device)

        bundle_ids = bundle_ids.to(bundle_logits.device).long()

        weight = None
        if self.bundle_class_weights is not None:
            weight = self.bundle_class_weights.to(bundle_logits.device)
        criterion_bundle = torch.nn.CrossEntropyLoss(weight=weight)

        return criterion_bundle(bundle_logits, bundle_ids)