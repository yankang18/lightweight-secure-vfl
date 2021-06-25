import numpy as np
import numpy.testing

from vnn.interacrive_dense_models import GuestDenseModel, EncryptedHostDenseModel
from vnn.interactive_layer import GuestInteractiveLayer, HostInteractiveLayer
from vnn.test.mock_local_model import MockLocalModel
from vnn.vnn import VerticalNeuralNetworkFederatedLearning


def get_input_data():
    x_g = np.array([[0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6]])
    x_h = np.array([[0.7, 0.8, 0.9],
                    [0.11, 0.12, 0.13]])
    return x_g, x_h


def get_dense_model_parameters():
    guest_parameters = {'weight': [[2, 3, 4]], 'bias': 1.5}
    host_parameters = {'weight': [[5, 6, 1]], 'bias': 2.5}
    return guest_parameters, host_parameters


def get_dense_models(input_shape, output_shape):
    guest_dense_model_parameters, host_dense_model_parameters = get_dense_model_parameters()
    guest_dense_model = GuestDenseModel()
    guest_dense_model.build(input_dim=input_shape, output_dim=output_shape, restore_stage=False)
    guest_dense_model.set_model_parameters(guest_dense_model_parameters)

    host_dense_model = EncryptedHostDenseModel()
    host_dense_model.build(input_dim=input_shape, output_dim=output_shape, restore_stage=False)
    host_dense_model.set_model_parameters(host_dense_model_parameters)
    return guest_dense_model, host_dense_model


if __name__ == "__main__":

    #
    # wire up models
    #

    host_local_model = MockLocalModel()
    guest_local_model = MockLocalModel()

    guest_dense_model, host_dense_model = get_dense_models(3, 1)

    guest_intr_layer = GuestInteractiveLayer(host_dense_model_dict={"A": host_dense_model},
                                             guest_dense_model=guest_dense_model,
                                             interactive_layer_lr=1.0)

    host_intr_layer = HostInteractiveLayer(host_id_set=set("A"),
                                           interactive_layer_lr=1.0)

    vnn = VerticalNeuralNetworkFederatedLearning(
        guest_local_model=guest_local_model,
        guest_top_learner=None,
        guest_interactive_layer=guest_intr_layer,
        host_local_model_dict={"A": host_local_model},
        host_interactive_layer=host_intr_layer,
        main_party_id="_main")

    #
    # prepare data
    #

    g_repr, h_repr = get_input_data()

    #
    # run forward computation
    #
    z_logit = vnn.forward_computation(g_repr, {"A": h_repr})

    #
    # prepare real values
    #
    guest_dense_model_parameters, host_dense_model_parameters = get_dense_model_parameters()

    guest_dense_model_weight = np.array(guest_dense_model_parameters['weight']).T
    guest_dense_model_bias = np.array(guest_dense_model_parameters['bias'])

    host_dense_model_weight = np.array(host_dense_model_parameters['weight']).T
    host_dense_model_bias = np.array(host_dense_model_parameters['bias'])

    g_logit = np.matmul(g_repr, guest_dense_model_weight)
    h_logit = np.matmul(h_repr, host_dense_model_weight)

    real_z_logit = g_logit + h_logit + guest_dense_model_bias
    print("g_logit:", g_logit)
    print("h_logit:", h_logit)
    print("guest_dense_model_bias:", guest_dense_model_bias)
    print("real_z_logit:", real_z_logit)
    print("z_logit:", z_logit)

    assert numpy.allclose(z_logit, real_z_logit), True
