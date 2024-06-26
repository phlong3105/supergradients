import os
from typing import Mapping, Tuple, Optional, Union, Any

import torch
import numpy as np
from torch import nn

from super_gradients.training.utils.checkpoint_utils import read_ckpt_state_dict
from super_gradients.training.utils.utils import move_state_dict_to_device, unwrap_model


class ModelWeightAveraging:
    """
    Utils class for managing the averaging of the best several snapshots into a single model.
    A snapshot dictionary file and the average model will be saved / updated at every epoch and evaluated only when
    training is completed. The snapshot file will only be deleted upon completing the training.
    The snapshot dict will be managed on cpu.
    """

    def __init__(
        self,
        ckpt_dir: str,
        greater_is_better: bool,
        metric_to_watch: str,
        load_checkpoint: bool = False,
        number_of_models_to_average: int = 10,
    ):
        """
        Init the ModelWeightAveraging
        :param ckpt_dir:                    The directory where the checkpoints are saved
        :param metric_to_watch:             Monitoring loss or acc, will be identical to that which determines best_model
        :param load_checkpoint:             Whether to load pre-existing snapshot dict.
        :param number_of_models_to_average: Number of models to average
        """

        self.averaging_snapshots_file = os.path.join(ckpt_dir, "averaging_snapshots.pkl")
        self.number_of_models_to_average = number_of_models_to_average
        self.metric_to_watch = metric_to_watch
        self.greater_is_better = greater_is_better

        # if continuing training, copy previous snapshot dict if exist
        if load_checkpoint and ckpt_dir is not None and os.path.isfile(self.averaging_snapshots_file):
            averaging_snapshots_dict = read_ckpt_state_dict(self.averaging_snapshots_file)
        else:
            averaging_snapshots_dict = {"snapshot" + str(i): None for i in range(self.number_of_models_to_average)}
            # if metric to watch is acc, hold a zero array, if loss hold inf array
            if self.greater_is_better:
                averaging_snapshots_dict["snapshots_metric"] = -1 * np.inf * np.ones(self.number_of_models_to_average)
            else:
                averaging_snapshots_dict["snapshots_metric"] = np.inf * np.ones(self.number_of_models_to_average)

            torch.save(averaging_snapshots_dict, self.averaging_snapshots_file)

    def update_snapshots_dict(self, model: nn.Module, validation_results_dict: Mapping[str, float]):
        """
        Update the snapshot dict and returns the updated average model for saving
        :param model: the latest model
        :param validation_results_dict: performance of the latest model
        """
        averaging_snapshots_dict = self._get_averaging_snapshots_dict()

        # IF CURRENT MODEL IS BETTER, TAKING HIS PLACE IN ACC LIST AND OVERWRITE THE NEW AVERAGE
        require_update, update_ind = self._is_better(averaging_snapshots_dict, validation_results_dict)
        if require_update:
            # moving state dict to cpu
            new_sd = unwrap_model(model).state_dict()
            new_sd = move_state_dict_to_device(new_sd, "cpu")

            averaging_snapshots_dict["snapshot" + str(update_ind)] = new_sd
            averaging_snapshots_dict["snapshots_metric"][update_ind] = float(validation_results_dict[self.metric_to_watch])

        return averaging_snapshots_dict

    def get_average_model(self, model, validation_results_dict=None) -> Mapping[str, torch.Tensor]:
        """
        Returns the averaged model
        :param model: will be used to determine arch
        :param validation_results_dict: if provided, will update the average model before returning
        :param target_device: if provided, return sd on target device

        """
        # If validation tuple is provided, update the average model
        if validation_results_dict is not None:
            averaging_snapshots_dict = self.update_snapshots_dict(model, validation_results_dict)
        else:
            averaging_snapshots_dict = self._get_averaging_snapshots_dict()

        torch.save(averaging_snapshots_dict, self.averaging_snapshots_file)
        average_model_sd = averaging_snapshots_dict["snapshot0"]
        for n_model in range(1, self.number_of_models_to_average):
            if averaging_snapshots_dict["snapshot" + str(n_model)] is not None:
                net_sd = averaging_snapshots_dict["snapshot" + str(n_model)]
                # USING MOVING AVERAGE
                for key in average_model_sd:
                    average_model_sd[key] = torch.true_divide(average_model_sd[key] * n_model + net_sd[key], (n_model + 1))

        return average_model_sd

    def cleanup(self):
        """
        Delete snapshot file when reaching the last epoch
        """
        os.remove(self.averaging_snapshots_file)

    def _is_better(
        self, averaging_snapshots_dict: Mapping[str, Any], validation_results_dict: Mapping[str, Union[float, torch.Tensor]]
    ) -> Tuple[bool, Optional[int]]:
        """
        Determines if the new model is better according to the specified metrics
        :param averaging_snapshots_dict: snapshot dict
        :param validation_results_dict:  latest model performance
        :return: Tuple (bool, index) whether first item is True if the new model is better and False otherwise;
                 Second item is the index in the averaging_snapshots_dict to which the new model should be saved
        """
        snapshot_metric_array = averaging_snapshots_dict["snapshots_metric"]
        val = float(validation_results_dict[self.metric_to_watch])

        if not np.isfinite(val):
            return False, None

        if self.greater_is_better:
            update_ind = np.argmin(snapshot_metric_array)
        else:
            update_ind = np.argmax(snapshot_metric_array)

        if (self.greater_is_better and val > snapshot_metric_array[update_ind]) or (not self.greater_is_better and val < snapshot_metric_array[update_ind]):
            return True, update_ind

        return False, None

    def _get_averaging_snapshots_dict(self):
        return torch.load(self.averaging_snapshots_file, map_location="cpu")
