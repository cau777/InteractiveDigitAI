class BatchConfig:
    def __init__(self, training: bool, version: int = 1):
        if version < 1:
            raise ValueError(f"Epoch should be greater than 0: {version}")
        self.epoch = version
        self.training = training
