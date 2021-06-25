class MockLocalModel(object):
    def __init__(self):
        pass

    def forward(self, x):
        return x

    def predict(self, x):
        pass

    def backward(self, x, grads):
        pass

    def get_output_dim(self):
        pass


class MockTopModel(object):
    def __init__(self):
        pass

    def forward(self, x):
        return x

    def predict(self, x):
        pass

    def backward(self, x, grads):
        pass

    def get_output_dim(self):
        pass

# class MockInteractiveLayerActivation(object):
#     def __init__(self, activation_func=None):
#         self.activation_func = activation_func
#         self.activation_input = None
#
#     def forward_activation(self, input_data):
#         # LOGGER.debug("[DEBUG] DenseModel.forward_activation")
#         self.activation_input = input_data
#         return self.activation_func(input_data) if self.activation_func else input_data
#
#     def backward_activation(self):
#         # LOGGER.debug("[DEBUG] DenseModel.backward_activation")
#         return torch.autograd.grad(outputs=self.activation_func, inputs=self.activation_input) if self.activation_func else 1.0
