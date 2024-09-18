import pytorch_lightning as pl
import abc
from torch import nn, Tensor
from torch.utils .data import DataLoader
from sklearn.model_selection import KFold, train_test_split
from utils.TorchExtensions import save_train_metrics


class BaseModel(pl.LightningModule, abc.ABC):
    def __init__(self, args):
        self.args = args
        super().__init__()

    def _init_params(self):
        """
        Apply Xavier uniform initialisation of learnable weights
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @abc.abstractmethod
    def encode_data(self, smiles) -> tuple[DataLoader, DataLoader, DataLoader]:
        raise NotImplementedError

    @abc.abstractmethod
    def encode_data_kfold(self, smiles) -> list:
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, src, trg) -> Tensor:
        raise NotImplementedError

    def on_fit_start(self) -> None:
        self.trainer.early_stopping_callback.patience = 10
        if self.trainer.early_stopping_callback.mode == "max":
            self.trainer.early_stopping_callback.best_score *= 1e-10
        else:
            self.trainer.early_stopping_callback.best_score *= 1e10

    @abc.abstractmethod
    def x_step(self, batch: tuple[Tensor, Tensor], step_type: str, save_output: bool = False) -> Tensor:
        raise NotImplementedError

    def training_step(self, batch: tuple[Tensor, Tensor], batch_id: int) -> Tensor:
        return self.x_step(batch, "train")

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_id: int) -> Tensor:
        return self.x_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        steps_per_epoch = {
            "train": self.trainer.num_training_batches if self.trainer.num_training_batches < float('inf') else 1,
            "val": self.trainer.num_val_batches[0],
            "test": self.trainer.num_val_batches[0]}
        save_train_metrics(self.metric_history, f"{self.__class__.__name__}/{self.trainer.logger.version}",
                           steps_per_epoch)

    @abc.abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError

    @abc.abstractmethod
    def reset_model(self):
        raise NotImplementedError
