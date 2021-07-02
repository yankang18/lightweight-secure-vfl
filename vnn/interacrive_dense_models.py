#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tempfile

import numpy as np
import torch
import torch.nn as nn
import time
from utils import get_logger

LOGGER = get_logger()


class InternalDenseModel(nn.Module):
    def __init__(self, input_dim, output_dim, apply_bias=True):
        super(InternalDenseModel, self).__init__()
        # LOGGER.debug(f"[DEBUG] InternalDenseModel with shape [{input_dim}, {output_dim}]")

        self.apply_bias = apply_bias
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim, bias=apply_bias),
        )

    def forward(self, x):
        return self.feature_extractor(x)

    def __set_parameters(self, weight, bias):
        def init_weights(m):
            if type(m) == nn.Linear:
                with torch.no_grad():
                    # print("weight:", weight)
                    # print("m.weight:", m.weight.shape)
                    m.weight.copy_(torch.tensor(weight, dtype=torch.double))
                    if bias:
                        m.bias.copy_(torch.tensor(bias, dtype=torch.double))

        self.feature_extractor.apply(init_weights)

    def set_parameters(self, parameters):
        weight = parameters["weight"]
        bias = parameters.get("bias") if self.apply_bias else None
        self.__set_parameters(weight, bias)

    def get_named_parameters(self):
        return self.feature_extractor.named_parameters()

    def get_parameters(self):
        return self.feature_extractor.parameters()

    def export_model(self):
        f = tempfile.TemporaryFile()
        try:
            torch.save(self.state_dict(), f)
            f.seek(0)
            model_bytes = f.read()
            return model_bytes
        finally:
            f.close()

    def restore_model(self, model_bytes):
        f = tempfile.TemporaryFile()
        f.write(model_bytes)
        f.seek(0)
        self.load_state_dict(torch.load(f))
        f.close()


class InteractiveLayerActivation(object):
    def __init__(self, activation_func=None):
        self.activation_func = activation_func
        self.activation_input = None

    def forward_activation(self, input_data):
        # LOGGER.debug("[DEBUG] DenseModel.forward_activation")
        self.activation_input = input_data
        return self.activation_func(input_data) if self.activation_func else input_data

    def backward_activation(self):
        # LOGGER.debug("[DEBUG] DenseModel.backward_activation")
        return torch.autograd.grad(outputs=self.activation_func,
                                   inputs=self.activation_input) if self.activation_func else 1.0


class BaseDenseModel(object):
    def __init__(self, role):
        self.role = role
        self.model_weight = None
        self.bias = 0
        self.internal_dense_model = None
        self.is_empty_model = False

        # only guest apply bias
        self.apply_bias = True if self.role == 'guest' else False
        # self.apply_bias = False

    def forward(self, x):
        pass

    def build(self, input_dim, output_dim, restore_stage=False):
        LOGGER.debug(f"Build [{self.role}] dense layer with input shape:{input_dim}, output shape:{output_dim}.")
        if not input_dim:
            if self.role == "host":
                raise ValueError("host input is empty!")
            else:
                self.is_empty_model = True
                return

        self.internal_dense_model = InternalDenseModel(input_dim=input_dim,
                                                       output_dim=output_dim,
                                                       apply_bias=self.apply_bias)

        if not restore_stage:
            self.__init_model_weight(self.internal_dense_model)

    def set_model_parameters(self, parameters):
        self.internal_dense_model.set_parameters(parameters)
        self.__init_model_weight(self.internal_dense_model)

    def export_model(self):
        if self.is_empty_model:
            return ''.encode()

        param = {"weight": self.model_weight.T}
        if self.apply_bias and self.bias is not None:
            param["bias"] = self.bias

        self.internal_dense_model.set_parameters(param)
        return self.internal_dense_model.export_model()

    def restore_model(self, model_bytes):
        if self.is_empty_model:
            return

        # LOGGER.debug("model_bytes is {}".format(model_bytes))
        self.internal_dense_model.restore_model(model_bytes)
        self.__init_model_weight(self.internal_dense_model)

    def __init_model_weight(self, model):
        # LOGGER.debug("BaseDenseModel._init_model_weight")
        model_params = [param.tolist() for param in model.parameters()]
        self.model_weight = np.array(model_params[0]).T
        # LOGGER.debug(f"weight: {self.model_weight}, {self.model_weight.shape}")
        if self.apply_bias:
            self.bias = np.array(model_params[1])
            # LOGGER.debug(f"bias: {self.bias}, {self.bias.shape}")

    def get_weight(self):
        return self.model_weight

    def get_bias(self):
        return self.bias

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    @property
    def empty(self):
        return self.is_empty_model

    @property
    def output_shape(self):
        return self.model_weight.shape[1:]


class GuestDenseModel(BaseDenseModel):

    def __init__(self, learning_rate):
        super(GuestDenseModel, self).__init__("guest")
        self.input = None
        self.learning_rate = learning_rate

    def forward(self, x):
        self.input = x
        # print("GuestDenseModel.forward")
        # print("x shape", x.shape)
        # print("model_weight shape", self.model_weight.shape)
        output = np.matmul(x, self.model_weight)
        if self.apply_bias and self.bias is not None:
            output += self.bias

        return output

    def get_input_gradient(self, delta):
        if self.empty:
            return None

        error = np.matmul(delta, self.model_weight.T)
        return error

    def get_weight_gradient(self, delta):
        if self.empty:
            return None

        # print("!!! GuestDenseModel.get_weight_gradient")
        # print("delta.T:", delta.T, delta.T.shape)
        # print("input:", self.input, self.input.shape)
        # delta_w = np.matmul(delta.T, self.input) / self.input.shape[0]
        delta_w = np.matmul(delta.T, self.input)
        return delta_w.T

    def update_weight(self, delta):
        # LOGGER.debug("GuestDenseModel.update_weight")
        # LOGGER.debug(f"delta:{delta},{delta.shape}")
        # LOGGER.debug(f"weight [before update]:{self.model_weight}, {self.model_weight.shape}")
        # LOGGER.debug(f"bias [before update]:{self.bias}, {self.bias.shape}")
        # LOGGER.debug(f"lr:{self.lr}")

        # self.model_weight -= self.lr * delta.T
        self.model_weight -= self.learning_rate * delta

        # LOGGER.debug(f"weight [after update]:{self.model_weight}, {self.model_weight.shape}")

    def update_bias(self, delta):
        # LOGGER.debug("GuestDenseModel.update_bias")
        # LOGGER.debug(f"delta:{delta},{delta.shape}")
        # LOGGER.debug(f"mean delta:{np.mean(delta)}")
        # LOGGER.debug(f"sum delta:{np.sum(delta)}")
        # LOGGER.debug(f"weight [before update]:{self.bias}, {self.bias.shape}")
        self.bias -= np.sum(delta, axis=0) * self.learning_rate
        # LOGGER.debug(f"weight [after update]:{self.bias}, {self.bias.shape}")


# class PlainHostDenseModel(BaseDenseModel):
#     def __init__(self):
#         super(PlainHostDenseModel, self).__init__("host")
#         self.input = None
#
#     def forward(self, x):
#         # print("[DEBUG] HostDenseModel.forward_dense")
#         """
#             x should be encrypted_host_input
#         """
#         self.input = x
#         output = np.matmul(x, self.model_weight)
#         if self.apply_bias and self.bias is not None:
#             output += self.bias
#
#         return output
#
#     def get_input_gradient(self, delta, acc_noise=None):
#         error = np.matmul(delta, self.model_weight.T)
#         return error
#
#     def get_weight_gradient(self, delta):
#         delta_w = np.matmul(delta.T, self.input)
#         return delta_w.T
#
#     def update_weight(self, delta):
#         # self.model_weight -= self.lr * delta.T
#         self.model_weight -= self.learning_rate * delta
#
#     def update_bias(self, delta):
#         self.bias -= np.sum(delta, axis=0) * self.learning_rate


class EncryptedHostDenseModel(BaseDenseModel):
    def __init__(self, learning_rate):
        super(EncryptedHostDenseModel, self).__init__("host")
        self.input = None
        self.learning_rate = learning_rate

    def forward(self, x):
        # LOGGER.debug("EncryptedHostDenseModel.forward_dense")
        """
            x should be encrypted host input
        """
        self.input = x
        output = np.matmul(x, self.model_weight)
        if self.apply_bias and self.bias is not None:
            output += self.bias

        return output

    def get_input_gradient(self, delta, acc_noise=None):
        # LOGGER.debug("EncryptedHostDenseModel.get_input_gradient")
        # print("delta:", delta.shape)
        # print("acc_noise:", acc_noise.shape)
        # start_time = time.time()
        if acc_noise is not None:
            error = np.matmul(delta, (self.model_weight + acc_noise).T)
        else:
            error = np.matmul(delta, self.model_weight.T)
        # end_time = time.time()
        # print(f"spend time : {end_time - start_time}")
        return error

    def get_weight_gradient(self, delta):
        # LOGGER.debug("EncryptedHostDenseModel.get_weight_gradient")
        # print("self.input:", self.input, self.input.shape)
        # print("delta:", delta.shape)
        # print("input:", self.input.shape)
        # start_time = time.time()
        delta_w = np.matmul(delta.T, self.input)
        # end_time = time.time()
        # print(f"spend time : {end_time - start_time}")
        return delta_w.T

    def update_weight(self, delta):
        # LOGGER.debug("EncryptedHostDenseModel.update_weight")
        # LOGGER.debug(f"weight [before update]:{self.model_weight}, {self.model_weight.shape}")
        # LOGGER.debug(f"delta:{delta.shape}")
        # LOGGER.debug(f"lr:{self.lr}")

        # start_time = time.time()
        # self.model_weight -= self.lr * delta.T
        self.model_weight -= self.learning_rate * delta
        # end_time = time.time()
        # print(f"spend time : {end_time - start_time}")

        # LOGGER.debug(f"weight [after update]:{self.model_weight}")

    def update_bias(self, delta):
        # LOGGER.debug("EncryptedHostDenseModel.update_bias")
        self.bias -= np.sum(delta, axis=0) * self.learning_rate
