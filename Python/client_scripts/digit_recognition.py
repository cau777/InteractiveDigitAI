import json

from codebase.integration import ClientInterfaceBase
from codebase.nn import NeuralNetworkController
from codebase.nn.layers import SequentialLayer, DenseLayer, ReshapeLayer, ConvolutionLayer, FlattenLayer, MaxPoolLayer
from codebase.nn.layers.activation import ReluLayer
from codebase.nn.loss_functions import CrossEntropyLossFunction
from codebase.nn.lr_optimizers import AdamLrOptimizer
from codebase.persistence import LazyList
from codebase.persistence.utils import load_compressed_classification


class ClientInterface(ClientInterfaceBase):
    def __init__(self, model_data: str, train_data: str, test_data: str):
        super().__init__()
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

        if model_data:
            model: dict[str] = json.loads(model_data)
            self.network.main_layer.set_trainable_params(model["params"])
            self.network.version = model["version"]

        if train_data:
            self.train_data = load_compressed_classification("mnist", ClientInterfaceBase.extract_bytes(train_data))

        if test_data:
            self.test_data = load_compressed_classification("mnist", ClientInterfaceBase.extract_bytes(test_data))

    def train(self, epochs: int):
        return self.network.train(self.train_data, epochs, measure=["avg_loss", "accuracy"])

    def test(self):
        return self.network.test(self.train_data, measure=["avg_loss", "accuracy"])

    def save(self):
        params: list[float] = self.network.main_layer.get_trainable_params()
        data = {"version": self.network.version,
                "params": params}
        return json.dumps(data)

    def benchmark(self):
        return self.network.benchmark((28, 28))


# noinspection PyUnresolvedReferences
instance = ClientInterface(model_data, train_data, test_data)
