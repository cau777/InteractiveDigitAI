import numpy as np
from libs.nn import NeuralNetworkController, TrainingExample
from libs.nn.layers import SequentialLayer
from libs.nn.loss_functions import MseLossFunction

n = NeuralNetworkController(SequentialLayer(), MseLossFunction())
n.evaluate_single(np.array([1, 2]))
