import numpy as np

from secureprotol.paillier import RandomNumberGenerator
from utils import get_logger
from vnn.encrypt_utils import PaillierEncryptHelper

LOGGER = get_logger()


class GuestInteractiveLayer(object):

    def __init__(self,
                 host_dense_model_dict,
                 guest_dense_model):

        self.rng_generator = RandomNumberGenerator()

        self.guest_dense_model = guest_dense_model
        self.host_dense_model_dict = host_dense_model_dict
        self.guest_side_forward_noise_dict = dict()
        self.guest_side_backward_noise_dict = dict()

    def compute_guest_logit(self, guest_repr):
        # LOGGER.trace("GUEST_FORWARD.compute_guest_logit")
        guest_logit = self.guest_dense_model.forward(guest_repr)
        return guest_logit

    def compute_encrypted_obfuscated_host_logit(self, host_encrypted_repr_dict):
        host_encrypted_obfuscated_logit_dict = dict()
        for host_id, host_encrypted_repr in host_encrypted_repr_dict.items():
            host_model = self.host_dense_model_dict[host_id]
            host_encrypted_obfuscated_logit = host_model.forward(host_encrypted_repr)
            rnd_noise = self.rng_generator.generate_random_number(host_encrypted_obfuscated_logit.shape)
            self.guest_side_forward_noise_dict[host_id] = rnd_noise
            host_encrypted_obfuscated_logit_dict[host_id] = host_encrypted_obfuscated_logit + rnd_noise
        return host_encrypted_obfuscated_logit_dict

    def compute_host_logit(self, host_logit_w_noise_dict):
        # LOGGER.trace("GUEST_FORWARD.compute_host_logit")

        host_logit_dict = dict()
        for host_id, host_logit_w_noise in host_logit_w_noise_dict.items():
            host_logit_dict[host_id] = host_logit_w_noise - self.guest_side_forward_noise_dict[host_id]
        return host_logit_dict

    def compute_comb_logit(self, host_logit_dict, guest_logit):
        # LOGGER.trace("GUEST_FORWARD.compute_activation")

        host_logit_list = [value for _, value in host_logit_dict.items()]
        comb_logit = host_logit_list[0]
        for idx in range(1, len(host_logit_list)):
            comb_logit += host_logit_list[idx]
        if self.guest_dense_model:
            comb_logit += guest_logit
        return comb_logit

    #
    # Guest forward complete
    #

    #
    # Guest backward start
    #
    def update_guest_dense_model(self, backward_gradient):
        guest_input_gradient = self.__update_guest(backward_gradient)
        return guest_input_gradient

    def compute_host_weight_gradient_w_noise(self, backward_gradient):
        # LOGGER.trace("GUEST_BACKWARD.compute_host_weight_gradient_w_noise")

        host_encrypted_weight_gradient_w_noise_dict = dict()
        for host_id, host_model in self.host_dense_model_dict.items():
            host_encrypted_weight_gradient = host_model.get_weight_gradient(backward_gradient)
            wg_noise = self.rng_generator.generate_random_number(host_encrypted_weight_gradient.shape)
            self.guest_side_backward_noise_dict[host_id] = wg_noise
            host_encrypted_weight_gradient_w_noise_dict[host_id] = host_encrypted_weight_gradient + wg_noise

        return host_encrypted_weight_gradient_w_noise_dict

    def update_host_dense_model(self, activation_grad, host_obfuscated_weight_grad_dict, enc_acc_noise_dict):
        host_input_grad_dict = dict()
        for host_id, host_obfuscated_weight_grad_w_noise in host_obfuscated_weight_grad_dict.items():
            host_obfuscated_weight_grad = host_obfuscated_weight_grad_w_noise - self.guest_side_backward_noise_dict[
                host_id]
            host_model = self.host_dense_model_dict[host_id]
            host_input_grad_dict[host_id] = host_model.get_input_gradient(activation_grad, enc_acc_noise_dict[host_id])
            self.__update_host(host_model, host_obfuscated_weight_grad)
        return host_input_grad_dict

    def __update_guest(self, activation_grad):
        input_gradient = self.guest_dense_model.get_input_gradient(activation_grad)
        weight_gradient = self.guest_dense_model.get_weight_gradient(activation_grad)
        self.guest_dense_model.update_weight(weight_gradient)
        self.guest_dense_model.update_bias(activation_grad)

        return input_gradient

    def __update_host(self, host_model, weight_grad):
        host_model.update_weight(weight_grad)


class HostInteractiveLayer(object):
    def __init__(self, host_id_set, interactive_layer_lr, close_encrypt=False):
        self.learning_rate = interactive_layer_lr
        self.rng_generator = RandomNumberGenerator()

        # state dictionaries
        self.encrypt_helper_dict = {host_id: PaillierEncryptHelper(close_encrypt=close_encrypt) for host_id in host_id_set}
        self.host_repr_dict = dict()
        self.acc_noise_dict = dict()
        self.init_acc_noise = True

    def get_acc_noise_dict(self):
        return self.acc_noise_dict

    #
    # Host forward start
    #
    def encrypt_host_repr(self, host_repr_dict):
        enc_host_repr_dict = dict()
        for host_id, host_repr in host_repr_dict.items():
            self.host_repr_dict[host_id] = host_repr
            enc_host_repr_dict[host_id] = self.encrypt_helper_dict[host_id].encrypt(input=host_repr)
        return enc_host_repr_dict

    def __get_acc_noise(self, host_id, model_output_unit):
        if self.init_acc_noise:
            self.init_acc_noise = False
            # accumulative noise is initialized to zero
            host_repr_dim = self.host_repr_dict[host_id].shape[1]
            acc_noise = np.zeros((host_repr_dim, model_output_unit))
            self.acc_noise_dict[host_id] = acc_noise
        else:
            acc_noise = self.acc_noise_dict[host_id]
        return acc_noise

    def compute_host_logit_w_noise(self, host_enc_obfuscated_logit_w_noise_dict):
        # LOGGER.trace("HOST_FORWARD.compute_host_logit_w_noise")
        host_logit_w_noise_dict = dict()
        for host_id, host_enc_obfuscated_logit_w_noise in host_enc_obfuscated_logit_w_noise_dict.items():
            host_obfuscated_logit_w_noise = self.encrypt_helper_dict[host_id].decrypt(
                input=host_enc_obfuscated_logit_w_noise)
            acc_noise = self.__get_acc_noise(host_id, host_obfuscated_logit_w_noise.shape[1])

            # LOGGER.trace(f"| host_obfuscated_logit_w_noise shape:{host_obfuscated_logit_w_noise.shape}.")
            # LOGGER.trace(f"| self.host_repr_dict[host_id] shape:{self.host_repr_dict[host_id].shape}.")
            # LOGGER.trace(f"| acc_noise shape:{acc_noise.shape}.")
            host_logit_w_noise_dict[host_id] = host_obfuscated_logit_w_noise + np.dot(self.host_repr_dict[host_id],
                                                                                      acc_noise)
        return host_logit_w_noise_dict

    #
    # Host forward complete
    #

    #
    # Host backward start
    #
    def compute_host_obfuscated_weight_gradient_w_noise(self, host_enc_weight_grad_w_noise_dict):
        # LOGGER.trace("HOST_BACKWARD.compute_host_obfuscated_weight_gradient_w_noise")
        enc_acc_noise_dict = dict()
        host_weight_gradient_w_noise_dict = dict()
        for host_id, host_enc_weight_grad_w_noise in host_enc_weight_grad_w_noise_dict.items():
            host_weight_grad_w_noise = self.encrypt_helper_dict[host_id].decrypt(host_enc_weight_grad_w_noise)

            weight_grad_noise = self.rng_generator.generate_random_number(host_weight_grad_w_noise.shape)
            host_weight_gradient_w_noise_dict[host_id] = host_weight_grad_w_noise + weight_grad_noise / self.learning_rate

            # print("encrypt_helper_dict:", self.encrypt_helper_dict)
            # print("acc_noise_dict:", self.acc_noise_dict)
            enc_acc_noise_dict[host_id] = self.encrypt_helper_dict[host_id].encrypt(self.acc_noise_dict[host_id])

            # accumulate noise
            # print("self.acc_noise_dict[host_id]:", self.acc_noise_dict[host_id], self.acc_noise_dict[host_id].shape)
            # print("weight_grad_noise:", weight_grad_noise, weight_grad_noise.shape)
            self.acc_noise_dict[host_id] = self.acc_noise_dict[host_id] + weight_grad_noise
            # print("self.acc_noise_dict[host_id]:", self.acc_noise_dict[host_id], self.acc_noise_dict[host_id].shape)

        return host_weight_gradient_w_noise_dict, enc_acc_noise_dict

    def decrypt_host_repr_gradient(self, host_enc_repr_gradient_dict):
        host_repr_dict = dict()
        for host_id, host_enc_repr_gradient in host_enc_repr_gradient_dict.items():
            host_repr_dict[host_id] = self.encrypt_helper_dict[host_id].decrypt(input=host_enc_repr_gradient)
        return host_repr_dict
    #
    # Host backward complete
    #
