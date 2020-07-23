from collections import Callable
from keras.layers import Layer

class CurveFit(Layer):
    def __init__(self,
                 parameters: int,     # the number of parameters
                                      # of our function
                 function: Callable,  # the function we want to fit
                 initializer='uniform', # how to initialize the
                                        # parameters
                 **kwargs):
        super().__init__(**kwargs)
        self.parameters = parameters
        self.function = function
        self.initializer = initializer

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.parameters, ),
                                      initializer=self.initializer,
                                      trainable=True)
        # Be sure to call this at the end
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # here we gonna invoke our function and return the result.
        # the loss function will do whatever is needed to fit this
        # function as good as possible by learning the parameters.
        return self.function(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        return input_shape
