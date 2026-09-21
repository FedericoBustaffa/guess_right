from __future__ import annotations

from guess_right.core import (
    BinaryCrossEntropy,
    CrossEntropy,
    Dataset,
    LeakyReLU,
    Linear,
    Loader,
    Loss,
    MeanSquaredError,
    Model,
    Module,
    Optimizer,
    Parameter,
    ReLU,
    Residual,
    Sigmoid,
    Softmax,
    StochasticGradientDescent,
    Tanh,
    Tensor,
    _no_grad_enabled,
    _set_no_grad,
    accuracy_score,
    cos,
    exp,
    log,
    matmul,
    maximum,
    mean,
    minimum,
    pow,
    product,
    round,
    sin,
    sum,
    tanh,
)

from . import core

__all__: list = [
    "Tensor",
    "Parameter",
    "Module",
    "Linear",
    "ReLU",
    "LeakyReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "Residual",
    "Model",
    "Dataset",
    "Loader",
    "Optimizer",
    "StochasticGradientDescent",
    "Loss",
    "MeanSquaredError",
    "BinaryCrossEntropy",
    "CrossEntropy",
    "accuracy_score",
    "sum",
    "mean",
    "product",
    "matmul",
    "pow",
    "log",
    "exp",
    "sin",
    "cos",
    "tanh",
    "maximum",
    "minimum",
    "round",
    "no_grad",
]

class no_grad:
    """
    Context manager equivalente alla guardia RAII NoGrad del C++.

        Esempio:
            with guess_right.no_grad():
                prediction = model(x)

    """
    def __enter__(self) -> no_grad: ...
    def __exit__(self, exc_type, exc_value, traceback) -> bool: ...
