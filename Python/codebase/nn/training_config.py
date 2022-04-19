class TrainingConfig:
    def __init__(self, epoch: int, batch_size: int):
        if epoch < 1:
            raise ValueError(f"Epoch should be greater than 0: {epoch}")
        self.epoch = epoch
        self.batch_size = batch_size
