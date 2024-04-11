# Read keras model into list of parameters like op, input, output, weight, bias
from __future__ import annotations
from dataclasses import dataclass
import typing
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer as KerasLayer
import numpy as np
import keras

supported_ops = [
    'Activation',
    'AveragePooling2D',
    'BatchNormalization',
    'Conv2D',
    'Dense',
    'Flatten',
    'GlobalAveragePooling2D',
    'GlobalMaxPooling2D',
    'MaxPooling2D',
    'ReLU',
    'Softmax',
    # edit
    'MeanCheck'
]

skip_ops = [
    'Dropout',
    'InputLayer',
]


# read each layer in a model and convert it to a class called Layer
@dataclass
class Layer:
    ''' A single layer in a Keras model. '''
    op: str
    name: str
    input: typing.List[int]
    output: typing.List[int]
    config: typing.Dict[str, typing.Any]
    weights: typing.List[np.ndarray]

    def __init__(self, layer: KerasLayer):
        self.op = layer.__class__.__name__
        self.name = layer.name
        # print('lllllayer: ', layer.input_shape)
        self.input = layer.input.shape[1:]
        self.output = layer.output.shape[1:]
        self.config = layer.get_config()
        self.weights = layer.get_weights()


# edit just skeleton to let this program recognize the layer, 
# doesnt really matter if it fits the definition, because we 
        # will just map layer_name --> circom template anyway
class MeanCheck(keras.layers.Layer):
    def __init__(self,*args, **kwargs ):
        super().__init__()
    def call(self, inputs):
        return keras.ops.sum(inputs, axis = 2)
    
class Model:
    layers: typing.List[Layer]

    def __init__(self, filename: str, raw: bool = False):
        ''' Load a Keras model from a file. '''
        # edit : allow reading customed layer, in this case MeanCheck
        keras.saving.get_custom_objects().clear()
        custom_objects = {"MeanCheck": MeanCheck}
        with keras.saving.custom_object_scope(custom_objects):
            model = load_model(filename)
        print("summ: ", model.summary())
        self.layers = [Layer(layer) for layer in model.layers if self._for_transpilation(layer.__class__.__name__)]
    
    @staticmethod
    def _for_transpilation(op: str) -> bool:
        if op in skip_ops:
            return False
        if op in supported_ops:
            return True
        raise NotImplementedError(f'Unsupported op: {op}')