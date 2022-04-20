from codebase.nn import NeuralNetworkController
from codebase.nn.layers import SequentialLayer, DenseLayer
from codebase.nn.loss_functions import MseLossFunction
from codebase.nn.lr_optimizers import ConstantLrOptimizer
from codebase.integration.console import ClientInterfaceBase
import numpy as np
import sys


class ClientInterface(ClientInterfaceBase):
    def __init__(self):
        self.n = NeuralNetworkController(SequentialLayer(
            DenseLayer.create_random(2, 8, ConstantLrOptimizer(), ConstantLrOptimizer()),
            DenseLayer.create_random(8, 2, ConstantLrOptimizer(), ConstantLrOptimizer())
        ), MseLossFunction())


console = ClientInterface().create_console()
