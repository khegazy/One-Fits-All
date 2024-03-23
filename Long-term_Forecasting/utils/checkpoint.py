import dataclasses
import logging
import os
import glob
import re
import json
import torch
from typing import Dict, List, Optional, Tuple

TensorDict = Dict[str, torch.Tensor]
Checkpoint = Dict[str, TensorDict]


@dataclasses.dataclass
class CheckpointState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR


class CheckpointBuilder:
    @staticmethod
    def create_checkpoint(state: CheckpointState) -> Checkpoint:
        state_dicts = {
            "model": state.model.state_dict(),
            "optimizer": state.optimizer.state_dict()
        }
        if state.lr_scheduler is None:
            state_dicts["lr_scheduler"] = None
        else: 
            state_dicts["lr_scheduler"] = state.lr_scheduler.state_dict()
        return state_dicts

    @staticmethod
    def load_checkpoint(
        state: CheckpointState, checkpoint: Checkpoint, strict: bool, model_only: bool = False
    ) -> None:
        state.model.load_state_dict(checkpoint["model"], strict=strict)  # type: ignore
        if not model_only:
            if state.optimizer is not None:
                state.optimizer.load_state_dict(checkpoint["optimizer"])
            if checkpoint["lr_scheduler"] is None:
                state.lr_scheduler = None
            else:
                state.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        else:
            print("Loading model only")


@dataclasses.dataclass
class CheckpointPathInfo:
    path: str
    tag: str
    epochs: int
    is_optimal: bool = True
    is_pretrain: bool = False


class CheckpointIO:
    def __init__(
        self, 
        directory: str,
        tag: str,
        keep: bool = False,
        is_pretrain: bool = None
    ) -> None:
        self.directory = directory
        self.tag = tag
        self.keep = keep
        self.is_pretrain = is_pretrain
        if is_pretrain is None:
            self.search_pretrain = False
            self.is_pretrain = False
        else:
            self.search_pretrain = True
        self.old_optimal_ckpt_path: Optional[str] = None
        self.old_latest_ckpt_path: Optional[str] = None
        self.old_optimal_train_values_path: Optional[str] = None
        self.old_latest_train_values_path: Optional[str] = None

        self._train_values_prefix = "train_values"
        self._epochs_string = "_epoch-"
        self._filename_extension = "pt"
        self._optimal_string = "_optimal"
        self._latest_string = "_latest"
        self._pretrain_string = "_pretrain"

    def _get_checkpoint_filename(
            self,
            epochs: int,
            is_optimal: bool = True,
            is_pretrain: bool = False
        ) -> str:
        train_type = self._pretrain_string if is_pretrain or self.is_pretrain else ""
        train_type += self._optimal_string if is_optimal else self._latest_string
        return (
            self.tag
            + train_type
            + self._epochs_string
            + str(epochs)
            + "."
            + self._filename_extension
        )
    
    def _get_train_values_filename(
        self, epochs: int, is_optimal: bool = True, is_pretrain: bool = False
    ) -> str:
        train_type = self._pretrain_string if is_pretrain or self.is_pretrain else ""
        train_type += self._optimal_string if is_optimal else self._latest_string
        return (
            self._train_values_prefix
            + train_type
            + self._epochs_string
            + str(epochs)
            + ".json"
        )

    def _list_file_paths(self) -> List[str]:
        if not os.path.isdir(self.directory):
            return []
        search_string = os.path.join(
            self.directory, f"*{self._filename_extension}"
        )
        all_paths = glob.glob(search_string)
        return [path for path in all_paths if os.path.isfile(path)]

    def _parse_checkpoint_path(self, path: str) -> Optional[CheckpointPathInfo]:
        filename = os.path.basename(path)
        is_optimal = self._latest_string not in filename
        is_pretrain = self._pretrain_string in filename
        train_type = self._pretrain_string if is_pretrain else ""
        train_type += self._optimal_string if is_optimal else self._latest_string
        
        regex = re.compile(
            rf"^(?P<tag>.+){self._epochs_string}(?P<epochs>\d+)\.{self._filename_extension}$"
        )
        match = regex.match(filename)

        return CheckpointPathInfo(
            path=path,
            tag=match.group("tag")[:-len(train_type)],
            epochs=int(match.group("epochs")),
            is_optimal=is_optimal,
            is_pretrain=is_pretrain
        )

    def _get_latest_checkpoint_path(self, is_optimal: bool = True, is_pretrain: bool = False) -> Optional[str]:
        all_file_paths = self._list_file_paths()
        checkpoint_info_list = [
            self._parse_checkpoint_path(path) for path in all_file_paths
        ]
        all_selected_checkpoint_info_list = [
            info for info in checkpoint_info_list\
                if info and info.tag == self.tag and is_optimal == info.is_optimal
        ]
        if len(all_selected_checkpoint_info_list) == 0:
            #logging.warning(
            print(
                f"Cannot find checkpoint with tag '{self.tag}' in '{self.directory}'"
            )
            return None, None

        if self.search_pretrain:
            selected_pretrain_checkpoint_info_list = [
                info for info in checkpoint_info_list if info.is_pretrain
            ]
        
        if not self.is_pretrain:
            selected_checkpoint_info_list = [
                info for info in checkpoint_info_list if not info.is_pretrain
            ]
        
        if not self.is_pretrain:
            if len(selected_checkpoint_info_list) > 0:
                latest_checkpoint_info = max(
                    selected_checkpoint_info_list, key=lambda info: info.epochs
                )
            elif self.search_pretrain:
                latest_checkpoint_info = max(
                    selected_pretrain_checkpoint_info_list, key=lambda info: info.epochs
                )
            else:
                print(
                    f"Cannot find checkpoint with tag '{self.tag}' in '{self.directory}'"
                )
                return None, None
        else:
            if len(selected_pretrain_checkpoint_info_list) > 0:
                latest_checkpoint_info = max(
                    selected_pretrain_checkpoint_info_list, key=lambda info: info.epochs
                )
            else:
                print(
                    f"Cannot find checkpoint with tag '{self.tag}' in '{self.directory}'"
                )
                return None, None

        # Get train values path
        if latest_checkpoint_info.path is None:
            latest_train_values_path = None
        else:
            latest_train_values_path = os.path.join(
                os.path.dirname(latest_checkpoint_info.path),
                self._get_train_values_filename(
                    latest_checkpoint_info.epochs,
                    latest_checkpoint_info.is_optimal,
                    latest_checkpoint_info.is_pretrain
                )
            )
        
        return latest_checkpoint_info.path, latest_train_values_path

    def save_checkpoint(
        self,
        checkpoint: Checkpoint,
        epochs: int,
        is_optimal: bool = True,
        keep_last: bool = False
    ) -> None:
        old_path = self.old_optimal_ckpt_path if is_optimal\
            else self.old_latest_ckpt_path
        if not self.keep and old_path and not keep_last:
            logging.debug(f"Deleting old checkpoint file: {old_path}")
            os.remove(old_path)

        filename = self._get_checkpoint_filename(epochs, is_optimal=is_optimal)
        path = os.path.join(self.directory, filename)
        logging.debug(f"Saving checkpoint: {path}")
        os.makedirs(self.directory, exist_ok=True)
        torch.save(obj=checkpoint, f=path)
        if is_optimal:
            self.old_optimal_ckpt_path = path
        else:
            self.old_latest_ckpt_path = path
            
    
    def save_train_values(
        self,
        epochs: int,
        train_steps: int,
        min_valid_loss: float,
        metrics: dict = None,
        is_optimal: bool = True,
        keep_last: bool = False
    ) -> None:
        # Remove previous file
        old_path = self.old_optimal_train_values_path if is_optimal\
            else self.old_latest_train_values_path
        if not self.keep and old_path and not keep_last:
            logging.debug(f"Deleting old train values file: {old_path}")
            os.remove(old_path)
        
        # Place values in dictionary to save as json
        data = {
            "epochs" : epochs,
            "train_steps" : train_steps,
            "min_valid_loss": min_valid_loss,
        }
        if metrics is not None:
            for k,v in metrics.items():
                if torch.is_tensor(v):
                    metrics[k] = v.detach().cpu().item()
            data["metrics"] = metrics
        
        # Saving
        path = os.path.join(
            self.directory, self._get_train_values_filename(epochs, is_optimal)
        )
        with open(path, 'w') as fp:
            json.dump(data, fp)
        if is_optimal:
            self.old_optimal_train_values_path = path
        else:
            self.old_latest_train_values_path = path

    def load_latest(
        self, is_optimal: bool = True, device: Optional[torch.device] = None
    ) -> Optional[Tuple[Checkpoint, int]]:
        ckpt_path, train_values_path = self._get_latest_checkpoint_path(
            is_optimal=is_optimal,
        )
        if ckpt_path is None:
            return None, None

        checkpoint, epochs = self.load_checkpoint(ckpt_path, device=device)
        train_values = self.load_train_values(train_values_path)
        assert(epochs == train_values["epochs"])
        return checkpoint, train_values

    def load_checkpoint(
        self, path: str, device: Optional[torch.device] = None
    ) -> Tuple[Checkpoint, int]:
        checkpoint_info = self._parse_checkpoint_path(path)

        if checkpoint_info is None:
            raise RuntimeError(f"Cannot find path '{path}'")

        print(f"Loading checkpoint: {checkpoint_info.path}")
        return (
            torch.load(f=checkpoint_info.path, map_location=device),
            checkpoint_info.epochs,
        )
    
    def load_train_values(
        self, path: str, metrics_device: Optional[torch.device] = None
    ) -> Tuple[Checkpoint, int]:
        if not os.path.exists(path):
            raise RuntimeError(f"Cannot find path '{path}'")

        print(f"Loading training values: {path}")
        with open(path, "r") as fp:
            return json.load(fp)


class CheckpointHandler:
    def __init__(self, *args, **kwargs) -> None:
        self.io = CheckpointIO(*args, **kwargs)
        self.builder = CheckpointBuilder()

    def save(
        self, 
        state: CheckpointState,
        epochs: int,
        train_steps: int,
        min_valid_loss: float,
        metrics: dict = None,
        is_optimal: bool = True,
        keep_last: bool = False
    ) -> None:
        checkpoint = self.builder.create_checkpoint(state)
        self.io.save_checkpoint(
            checkpoint, epochs, is_optimal=is_optimal, keep_last=keep_last
        )
        self.io.save_train_values(
            epochs,
            train_steps,
            metrics=metrics,
            min_valid_loss=min_valid_loss,
            is_optimal=is_optimal,
            keep_last=keep_last
        )

    def load_latest(
        self,
        state: CheckpointState,
        device: Optional[torch.device] = None,
        strict=False,
        is_optimal=True,
        is_expected=False
    ) -> Optional[int]:
        checkpoint, train_values = self.io.load_latest(
            device=device, is_optimal=is_optimal
        )
        if checkpoint is None:
            if is_expected:
                raise ImportError("Cannot load expected checkpoint")
            else:
                print("Model is starting from scratch")
                return None

        load_model_only = self.io.search_pretrain and not self.io.is_pretrain
        self.builder.load_checkpoint(
            state=state,
            checkpoint=checkpoint,
            strict=strict,
            model_only=load_model_only
        )
        train_values = None if load_model_only else train_values
        return train_values

    def load_checkpoint(
        self,
        state: CheckpointState,
        path: str,
        strict=False,
        device: Optional[torch.device] = None,
    ) -> int:
        checkpoint, epochs = self.io.load_checkpoint(path, device=device)
        self.builder.load_checkpoint(
            state=state, checkpoint=checkpoint, strict=strict
        )
        return epochs