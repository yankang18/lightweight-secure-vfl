import numpy
import numpy as np
import torch
import torch.optim as optim

from vnn.interacrive_dense_models import InteractiveLayerActivation, InternalDenseModel
from vnn.interactive_layer import GuestInteractiveLayer, HostInteractiveLayer
from vnn.test.mock_local_model import MockLocalModel
from vnn.test.test_interactive_forward import get_dense_models, get_input_data, get_dense_model_parameters
from vnn.vnn import VerticalNeuralNetworkFederatedLearning

if __name__ == "__main__":
    # wire up models
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

    # prepare data
    g_repr, h_repr = get_input_data()
    activation_gradient = np.array([[-4.1028e-07],
                                    [1.8272e-04]], dtype=np.double)
    real_z_logit = vnn.forward_computation(g_repr, {"A": h_repr})

    # run backward computation
    guest_repr_grad, host_repr_grad_dict = vnn.backward_computation(activation_grad=activation_gradient)

    # prepare target
    guest_dense_model_parameters, host_dense_model_parameters = get_dense_model_parameters()
    t_guest_dense_model = InternalDenseModel(input_dim=3, output_dim=1, apply_bias=True)
    t_guest_dense_model.set_parameters(guest_dense_model_parameters)
    t_host_dense_model = InternalDenseModel(input_dim=3, output_dim=1, apply_bias=False)
    t_host_dense_model.set_parameters(host_dense_model_parameters)

    g_repr_tensor = torch.tensor(g_repr, dtype=torch.float, requires_grad=True)
    h_repr_tensor = torch.tensor(h_repr, dtype=torch.float, requires_grad=True)
    guest_logit = t_guest_dense_model.forward(g_repr_tensor)
    host_logit = t_host_dense_model.forward(h_repr_tensor)
    target_z_logit = guest_logit + host_logit

    optimizer = optim.SGD(list(t_guest_dense_model.parameters()) + list(t_host_dense_model.parameters()), lr=1.0)
    activation_gradient_tensor = torch.tensor(activation_gradient, dtype=torch.double)
    target_z_logit.backward(gradient=activation_gradient_tensor, retain_graph=True)
    optimizer.step()

    assert numpy.allclose(real_z_logit, target_z_logit.detach().numpy()), "Failed!"

    actual_guest_dense_weight = guest_dense_model.get_weight()
    actual_guest_dense_bias = guest_dense_model.get_bias()
    print(f"####> actual guest dense weight: {actual_guest_dense_weight}")
    print(f"####> actual guest dense bias: {actual_guest_dense_bias}")
    acc_noise_dict = host_intr_layer.get_acc_noise_dict()
    actual_host_dense_weight = host_dense_model.get_weight() + acc_noise_dict['A']
    print(f"####> actual host dense weight:{actual_host_dense_weight}")

    t_guest_dense_model_params = list(t_guest_dense_model.get_parameters())
    t_guest_dense_model_weight = t_guest_dense_model_params[0].detach().numpy().T
    t_guest_dense_model_bias = t_guest_dense_model_params[1].detach().numpy()
    t_host_dense_model_weight = list(t_host_dense_model.get_parameters())[0].detach().numpy().T
    print("####> target guest dense weight:", t_guest_dense_model_weight)
    print("####> target guest dense bias:", t_guest_dense_model_bias)
    print("####> target host dense weight:\n", t_host_dense_model_weight)

    assert numpy.allclose(actual_guest_dense_weight, t_guest_dense_model_weight), "Failed!"
    assert numpy.allclose(actual_guest_dense_bias, t_guest_dense_model_bias), "Failed!"
    assert numpy.allclose(actual_host_dense_weight, t_host_dense_model_weight), "Failed!"

    print("####> g_repr_tensor grad:", g_repr_tensor.grad.numpy())
    print("####> h_repr_tensor grad:", h_repr_tensor.grad.numpy())
    print("####> guest_repr_grad:\n", guest_repr_grad)
    print("####> host_repr_grad_dict:\n", host_repr_grad_dict)

    assert numpy.allclose(guest_repr_grad, g_repr_tensor.grad.numpy()), "Failed!"
    assert numpy.allclose(host_repr_grad_dict['A'], h_repr_tensor.grad.numpy()), "Failed!"
