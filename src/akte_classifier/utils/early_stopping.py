from typing import Optional

from loguru import logger


class EarlyStopping:
    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0.0,
    ) -> None:
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss
            improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as
            an improvement. Default: 0.0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.early_stop = False
        self.delta = delta
        self.improved = False

    def __call__(self, val_loss: float) -> None:
        self.improved = False
        if self.best_loss is None:
            self.best_loss = val_loss
            self.improved = True
        elif val_loss > self.best_loss - self.delta:
            # Loss did not improve enough (needs to be lower than best - delta)
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            improved_by = self.best_loss - val_loss
            logger.success(f"Validation loss improved by: {improved_by:.5f}")
            self.best_loss = val_loss
            self.improved = True
            self.counter = 0
