from vnn.interacrive_dense_models import GuestDenseModel, EncryptedHostDenseModel
from vnn.interacrive_dense_models import InteractiveLayerActivation
from vnn.interactive_layer import GuestInteractiveLayer, HostInteractiveLayer
from vnn.test.mock_local_model import MockTopModel
from vnn.vnn import VerticalNeuralNetworkFederatedLearning
import numpy as np
import numpy.testing

import torch
import torch.nn as nn
import torch.nn


if __name__ == "__main__":

    # wire up models
    guest_top_model = MockTopModel()

    activation_fn = nn.Sigmoid()
    intr_activation = InteractiveLayerActivation(activation_func=activation_fn)

    vnn = VerticalNeuralNetworkFederatedLearning(
        guest_local_model=None,
        guest_top_learner=guest_top_model,
        guest_interactive_layer=None,
        host_local_model_dict=None,
        host_interactive_layer=None,
        main_party_id="_main")

    # prepare data
    z_logit = np.array([[12.7], [7.6]])
    label = np.array([1, 0])

    # run forward computation
    label_tensor = torch.tensor(label, dtype=torch.float)
    label_tensor = label_tensor.reshape(-1, 1)
    activation_gradient, loss = vnn.train_top(z_logit, label_tensor)

    print("activation_gradient:", activation_gradient)
    print("loss:", loss)

    # # real values
    # guest_dense_model_weight = np.array(guest_dense_model_parameters['weight']).T
    # guest_dense_model_bias = np.array(guest_dense_model_parameters['bias'])
    #
    # host_dense_model_weight = np.array(host_dense_model_parameters['weight']).T
    # host_dense_model_bias = np.array(host_dense_model_parameters['bias'])
    #
    # g_logit = np.matmul(x_g, guest_dense_model_weight)
    # h_logit = np.matmul(x_h, host_dense_model_weight)
    #
    # z_logit = g_logit + h_logit + guest_dense_model_bias
    # print("g_logit:", g_logit)
    # print("h_logit:", h_logit)
    # print("guest_dense_model_bias:", guest_dense_model_bias)
    # print("z_logit:", z_logit)
    # print("activation:", activation)
    #
    # assert numpy.allclose(z_logit, activation), True
