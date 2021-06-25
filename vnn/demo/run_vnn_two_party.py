import torch.nn as nn
from sklearn.utils import shuffle

from datasets.nus_wide_dataset import load_two_party_data
from vlr.pytorch_models import LocalModel
from vlr.vfl_fixture import FederatedLearningFixture
from vnn.interacrive_dense_models import GuestDenseModel, EncryptedHostDenseModel
from vnn.interactive_layer import GuestInteractiveLayer, HostInteractiveLayer
from vnn.models.guest_top_model import GuestTopModelLearner, GuestTopModel
from vnn.vnn import VerticalNeuralNetworkFederatedLearning


def get_dense_models(input_dim, output_dim, learning_rate):
    guest_dense_model = GuestDenseModel(learning_rate=learning_rate)
    guest_dense_model.build(input_dim=input_dim, output_dim=output_dim, restore_stage=False)

    host_dense_model = EncryptedHostDenseModel(learning_rate=learning_rate)
    host_dense_model.build(input_dim=input_dim, output_dim=output_dim, restore_stage=False)
    return guest_dense_model, host_dense_model


def get_top_mode_learner(top_model_input_dim, optim_dict):
    top_model = GuestTopModel(input_dim=top_model_input_dim)

    top_learner = GuestTopModelLearner(
        top_model=top_model,
        classifier_criterion=nn.BCEWithLogitsLoss(),
        optim_dict=optim_dict,
        logit_activation_fn=None)

    return top_learner


def run_experiment(train_data, test_data, batch_size, epoch):
    print("hyper-parameters:")
    print("batch size: {0}".format(batch_size))

    Xa_train, Xb_train, y_train = train_data
    Xa_test, Xb_test, y_test = test_data

    intr_layer_learning_rate = 0.001
    host_local_optimizer_dict = {'learning_rate': 0.001, 'momentum': 0.99, 'weight_decay': 0.0001}
    guest_local_optimizer_dict = {'learning_rate': 0.001, 'momentum': 0.99, 'weight_decay': 0.0001}
    top_optim_dict = {'learning_rate': 0.001, 'momentum': 0.99, 'weight_decay': 0.0001}

    close_encrypt = True
    is_trace = False

    print("################################ Wire Federated Models ############################")

    guest_local_model_dim = 60
    host_local_model_dim = 60

    dense_model_input_dim = 60
    dense_model_out_dim = 30

    top_model_input_dim = 30

    # create local models for guest and host parties.
    guest_local_model = LocalModel(input_dim=Xa_train.shape[1], output_dim=guest_local_model_dim,
                                   optimizer_dict=guest_local_optimizer_dict)
    host_local_model = LocalModel(input_dim=Xb_train.shape[1], output_dim=host_local_model_dim,
                                  optimizer_dict=host_local_optimizer_dict)
    party_host_id = 'A'

    # create dense models of interactive layer for both guest repr and host repr.
    guest_dense_model, host_dense_model = get_dense_models(dense_model_input_dim,
                                                           dense_model_out_dim,
                                                           intr_layer_learning_rate)
    guest_intr_layer = GuestInteractiveLayer(host_dense_model_dict={"A": host_dense_model},
                                             guest_dense_model=guest_dense_model)

    host_intr_layer = HostInteractiveLayer(host_id_set=set(party_host_id),
                                           interactive_layer_lr=intr_layer_learning_rate,
                                           close_encrypt=close_encrypt)

    # create top model
    top_learner = get_top_mode_learner(top_model_input_dim=top_model_input_dim, optim_dict=top_optim_dict)

    # create vertical neural network federated learning
    federated_learning = VerticalNeuralNetworkFederatedLearning(
        guest_local_model=guest_local_model,
        guest_top_learner=top_learner,
        guest_interactive_layer=guest_intr_layer,
        host_local_model_dict={party_host_id: host_local_model},
        host_interactive_layer=host_intr_layer,
        main_party_id="_main")
    federated_learning.set_trace(is_trace=is_trace)

    print("################################ Train Federated Models ############################")

    fl_fixture = FederatedLearningFixture(federated_learning)

    # only guest party has labels (i.e., Y), host parties only have features (e.g., X).
    # 'party_list' stores X for all other parties.
    # Since this is two-party VFL, 'party_list' only stores the X of host party.
    train_data = {federated_learning.get_main_party_id(): {"X": Xa_train, "Y": y_train},
                  "party_list": {party_host_id: Xb_train}}
    test_data = {federated_learning.get_main_party_id(): {"X": Xa_test, "Y": y_test},
                 "party_list": {party_host_id: Xb_test}}

    fl_fixture.fit(train_data=train_data, test_data=test_data, epochs=epoch, batch_size=batch_size)


if __name__ == '__main__':
    print("################################ Prepare Data ############################")
    # TODO: change the data directory to [your data directory]
    data_dir = "../../../../../Data/"

    class_lbls = ['person', 'animal']
    train, test = load_two_party_data(data_dir, class_lbls, neg_label=0)
    Xa_train, Xb_train, y_train = train
    Xa_test, Xb_test, y_test = test

    batch_size = 128
    epoch = 3

    Xa_train, Xb_train, y_train = shuffle(Xa_train, Xb_train, y_train)
    Xa_test, Xb_test, y_test = shuffle(Xa_test, Xb_test, y_test)
    train = [Xa_train, Xb_train, y_train]
    test = [Xa_test, Xb_test, y_test]
    run_experiment(train_data=train, test_data=test, batch_size=batch_size, epoch=epoch)
