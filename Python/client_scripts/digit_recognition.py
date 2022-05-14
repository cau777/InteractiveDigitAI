import numpy as np

from codebase.integration import ClientInterfaceBase
from codebase.nn import NeuralNetworkController
from codebase.nn.layers import SequentialLayer, DenseLayer, ReshapeLayer, ConvolutionLayer, FlattenLayer, MaxPoolLayer
from codebase.nn.layers.activation import ReluLayer
from codebase.nn.loss_functions import CrossEntropyLossFunction
from codebase.nn.lr_optimizers import AdamLrOptimizer
from codebase.persistence.utils import load_compressed_classification


class ClientInterface(ClientInterfaceBase):
    def __init__(self):
        super().__init__()
        self.train_data = None
        self.test_data = None
        self.loaded_hash: int | None = None
        self.params = dict()

        self.network = NeuralNetworkController(SequentialLayer(
            ReshapeLayer((1, 28, 28)),

            ConvolutionLayer.create_random(1, 32, 3, AdamLrOptimizer(0.01)),
            ReluLayer(),
            MaxPoolLayer(2, 2),

            FlattenLayer(),

            DenseLayer.create_random(5408, 100, AdamLrOptimizer(), AdamLrOptimizer()),
            ReluLayer(),
            DenseLayer.create_random(100, 10, AdamLrOptimizer(), AdamLrOptimizer())
        ), CrossEntropyLossFunction())

    def train(self):
        # epochs: int
        return self.network.train(self.train_data, self.params["epochs"], measure=["avg_loss", "accuracy"])

    def test(self):
        return self.network.test(self.test_data, measure=["avg_loss", "accuracy"])

    def save(self):
        params: list[float] = self.network.main_layer.get_trainable_params()
        data = {"version": self.network.version,
                "params": params}
        return data

    def should_load(self):
        # hash: str
        return self.loaded_hash != self.params["hash"]

    def load(self):
        # version: int
        # params: list[float]
        # hash: str
        self.network.version = self.params["version"]
        self.loaded_hash = self.params["hash"]
        self.network.main_layer.set_trainable_params(map(float, self.params["params"]))

    def eval(self):
        # inputs: list[float]
        result = self.network.classify_single(np.array(self.params["inputs"]).reshape([28, 28]))
        print(result)
        return int(result)

    def benchmark(self):
        return self.network.benchmark((28, 28))

    def load_train_set(self):
        # data: str
        self.train_data = load_compressed_classification("mnist", ClientInterfaceBase.extract_bytes(self.params["data"]))

    def load_test_set(self):
        # data: str
        self.test_data = load_compressed_classification("mnist", ClientInterfaceBase.extract_bytes(self.params["data"]))


instance = ClientInterface()
