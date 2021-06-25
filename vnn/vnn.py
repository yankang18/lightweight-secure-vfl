from utils import get_logger

LOGGER = get_logger()

import time

class VerticalNeuralNetworkFederatedLearning(object):

    def __init__(self,
                 guest_local_model,
                 guest_top_learner,
                 guest_interactive_layer,
                 host_local_model_dict,
                 host_interactive_layer,
                 main_party_id="_main"):
        super(VerticalNeuralNetworkFederatedLearning, self).__init__()
        self.main_party_id = main_party_id

        self.guest_top_learner = guest_top_learner
        self.guest_local_model = guest_local_model
        self.guest_intr_layer = guest_interactive_layer

        self.host_local_model_dict = host_local_model_dict
        self.host_intr_layer = host_interactive_layer

        self.is_trace = False

    def set_trace(self, is_trace):
        self.is_trace = is_trace

    def get_main_party_id(self):
        return self.main_party_id

    def train_top(self, z_logit, y):
        """

        :param z_logit: numpy
        :param y:
        :return: activation gradient, numpy
        """
        if self.is_trace: LOGGER.trace("---------- VNN_TRAIN_TOP ----------")
        backward_gradient, loss = self.guest_top_learner.train_top(z_logit, y)
        return backward_gradient, loss

    def forward_computation(self, guest_repr, host_repr_dict):
        if self.is_trace: LOGGER.trace("---------- VNN_FORWARD_COMPUTATION ----------")

        guest_logit = self.guest_intr_layer.compute_guest_logit(guest_repr)
        # print("[TRACE] guest_logit", guest_logit, guest_logit.shape)

        enc_host_repr_dict = self.host_intr_layer.encrypt_host_repr(host_repr_dict)
        # all host interact with guest and obtain all host_logit in plain text
        host_enc_obfuscated_logit_dict = self.guest_intr_layer.compute_encrypted_obfuscated_host_logit(
            enc_host_repr_dict)
        host_logit_w_noise_dict = self.host_intr_layer.compute_host_logit_w_noise(host_enc_obfuscated_logit_dict)
        host_logit_dict = self.guest_intr_layer.compute_host_logit(host_logit_w_noise_dict)

        # print("VNN_FORWARD_COMPUTATION guest_logit:", guest_logit)
        # print("VNN_FORWARD_COMPUTATION host_logit_dict:", host_logit_dict)
        # guest compute activation of combination of host_logits and guest logit (output of interactive layer)
        return self.guest_intr_layer.compute_comb_logit(host_logit_dict, guest_logit)

    def backward_computation(self, activation_grad):
        if self.is_trace: LOGGER.trace("---------- VNN_BACKWARD_COMPUTATION ----------")

        # guest locally compute gradient of activation and gradient of guest_repr
        guest_repr_grad = self.guest_intr_layer.update_guest_dense_model(activation_grad)

        # all host interact with guest and obtain all host_logit in plain text
        host_enc_weight_grad_w_noise_dict = self.guest_intr_layer.compute_host_weight_gradient_w_noise(activation_grad)
        result = self.host_intr_layer.compute_host_obfuscated_weight_gradient_w_noise(host_enc_weight_grad_w_noise_dict)
        host_weight_grad_w_noise_dict, enc_acc_noise_dict = result
        host_enc_repr_grad_dict = self.guest_intr_layer.update_host_dense_model(activation_grad,
                                                                                host_weight_grad_w_noise_dict,
                                                                                enc_acc_noise_dict)

        host_repr_grad_dict = self.host_intr_layer.decrypt_host_repr_gradient(host_enc_repr_grad_dict)

        return guest_repr_grad, host_repr_grad_dict

    def compute_local_repr(self, x_g, host_x_dict):
        if self.is_trace: LOGGER.trace(f"---------- VNN_COMPUTE_LOCAL_REPR----------")
        guest_repr = self.guest_local_model.forward(x_g)
        host_repr_dict = {host_id: self.host_local_model_dict[host_id].forward(h_x) for host_id, h_x in
                          host_x_dict.items()}
        return guest_repr, host_repr_dict

    def fit(self, x_g, y, host_x_dict, global_step):
        if self.is_trace: LOGGER.trace(f"========== VNN_FIT {global_step} ==========")
        start_time = time.time()
        guest_repr, host_repr_dict = self.compute_local_repr(x_g, host_x_dict)
        end_time = time.time()
        if self.is_trace: LOGGER.trace(f"local repr computation spend:{end_time - start_time}")

        start_time = time.time()
        z_logit = self.forward_computation(guest_repr, host_repr_dict)
        end_time = time.time()
        if self.is_trace: LOGGER.trace(f"forward computation spend:{end_time - start_time}")

        start_time = time.time()
        backward_grad, loss = self.train_top(z_logit, y)
        end_time = time.time()
        if self.is_trace: LOGGER.trace(f"train top model spend:{end_time - start_time}")

        start_time = time.time()
        guest_repr_grad, host_repr_grad_dict = self.backward_computation(backward_grad)
        end_time = time.time()
        if self.is_trace: LOGGER.trace(f"backward computation spend:{end_time - start_time}")

        start_time = time.time()
        if self.is_trace: LOGGER.trace(f"---------- VNN_UPDATE_LOCAL_MODELS----------")
        self.guest_local_model.backward(x_g, guest_repr_grad)
        for host_id, host_local_model in self.host_local_model_dict.items():
            host_local_model.backward(host_x_dict[host_id], host_repr_grad_dict[host_id])
        end_time = time.time()
        if self.is_trace: LOGGER.trace(f"update local model spend:{end_time - start_time}")

        return loss

    def predict(self, x_g, host_x_dict):
        guest_repr, host_repr_dict = self.compute_local_repr(x_g, host_x_dict)
        z_logit = self.forward_computation(guest_repr, host_repr_dict)
        return self.guest_top_learner.predict(z_logit)
