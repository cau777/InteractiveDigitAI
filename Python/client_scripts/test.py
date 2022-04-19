from codebase.nn.utils import np
from codebase.nn import NeuralNetworkController, TrainingExample
from codebase.nn.layers import SequentialLayer, DenseLayer
from codebase.nn.lr_optimizers import ConstantLrOptimizer
from codebase.nn.loss_functions import MseLossFunction

n = NeuralNetworkController(SequentialLayer(
    DenseLayer.create_random(2, 8, ConstantLrOptimizer(), ConstantLrOptimizer()),
    DenseLayer.create_random(8, 2, ConstantLrOptimizer(), ConstantLrOptimizer())
), MseLossFunction())
n.evaluate_single(np.array([1, 2]))
