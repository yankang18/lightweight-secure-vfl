import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader
# from datasets.nus_wide_dataset import load_two_party_data
from dataset import MultiViewDataset6Party
from vlr.pytorch_models import LocalModel
from vlr.vfl_fixture import FederatedLearningFixture
from vnn.interacrive_dense_models import GuestDenseModel, EncryptedHostDenseModel
from vnn.interactive_layer import GuestInteractiveLayer, HostInteractiveLayer
from vnn.models.guest_top_model import GuestTopModelLearner, GuestTopModel
from vnn.vnn import VerticalNeuralNetworkFederatedLearning
import argparse
import time
import utils
import logging
import os
import sys
import glob

parser = argparse.ArgumentParser("modelnet40v1png")
# parser.add_argument('--data', required=True, help='location of the data corpus')
parser.add_argument('--name', type=str, required=True, help='experiment name')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--u_dim', type=int, default=64, help='u layer dimensions')
parser.add_argument('--k', type=int, required=True, help='num of client')

args = parser.parse_args()
args.name = '../../experiments/{}-{}'.format(args.name, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.name, scripts_to_save=glob.glob('*/*.py') + glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.name, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("args = %s", args)


def get_dense_models(input_dim, output_dim, learning_rate, host_id_set):
    guest_dense_model = GuestDenseModel(learning_rate=learning_rate)
    guest_dense_model.build(input_dim=input_dim, output_dim=output_dim, restore_stage=False)
    host_dense_model_dict = {}
    for host_id in host_id_set:
        host_dense_model_dict[host_id] = EncryptedHostDenseModel(learning_rate=learning_rate)
        host_dense_model_dict[host_id].build(input_dim=input_dim, output_dim=output_dim, restore_stage=False)
    return guest_dense_model, host_dense_model_dict


def get_top_mode_learner(top_model_input_dim, optim_dict):
    top_model = GuestTopModel(input_dim=top_model_input_dim)

    top_learner = GuestTopModelLearner(
        top_model=top_model,
        classifier_criterion=nn.CrossEntropyLoss(),
        optim_dict=optim_dict,
        logit_activation_fn=None)

    return top_learner


def run_experiment(train_data, test_data, batch_size, epoch):
    print("hyper-parameters:")
    print("batch size: {0}".format(batch_size))

    # Xa_train, Xb_train, y_train = train_data
    # Xa_test, Xb_test, y_test = test_data

    intr_layer_learning_rate = args.learning_rate
    host_local_optimizer_dict = {'learning_rate': args.learning_rate, 'momentum': 0.99, 'weight_decay': 0.0001}
    guest_local_optimizer_dict = {'learning_rate': args.learning_rate, 'momentum': 0.99, 'weight_decay': 0.0001}
    top_optim_dict = {'learning_rate': args.learning_rate, 'momentum': 0.99, 'weight_decay': 0.0001}

    close_encrypt = True
    is_trace = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    # device = 'cpu'
    print("################################ Wire Federated Models ############################")

    guest_local_model_dim = args.u_dim
    host_local_model_dim = args.u_dim

    dense_model_input_dim = args.u_dim
    dense_model_out_dim = args.u_dim

    top_model_input_dim = args.u_dim
    party_host_id = ['A', 'B', 'C', 'D', 'E'][:args.k - 1]
    logging.info("party_host_id: {}".format(" ".join(party_host_id)))
    # party_host_id = ['A']
    host_local_model_dict = {}
    # create local models for guest and host parties.
    guest_local_model = LocalModel(input_dim=32, output_dim=guest_local_model_dim,
                                   optimizer_dict=guest_local_optimizer_dict, device=device)
    for id in set(party_host_id):
        host_local_model_dict[id] = LocalModel(input_dim=32, output_dim=host_local_model_dim,
                                               optimizer_dict=host_local_optimizer_dict, device=device)

    # create dense models of interactive layer for both guest repr and host repr.
    guest_dense_model, host_dense_model_dict = get_dense_models(dense_model_input_dim,
                                                                dense_model_out_dim,
                                                                intr_layer_learning_rate,
                                                                set(party_host_id))
    guest_intr_layer = GuestInteractiveLayer(host_dense_model_dict=host_dense_model_dict,
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
        host_local_model_dict=host_local_model_dict,
        host_interactive_layer=host_intr_layer,
        main_party_id="_main")
    federated_learning.set_trace(is_trace=is_trace)

    print("################################ Train Federated Models ############################")

    fl_fixture = FederatedLearningFixture(federated_learning, logging)

    # only guest party has labels (i.e., Y), host parties only have features (e.g., X).
    # 'party_list' stores X for all other parties.
    # Since this is two-party VFL, 'party_list' only stores the X of host party.
    train_data = {federated_learning.get_main_party_id(): {"X": train_data, "Y": train_data},
                  "party_list": set(party_host_id)}
    test_data = {federated_learning.get_main_party_id(): {"X": test_data, "Y": test_data},
                 "party_list": set(party_host_id)}

    fl_fixture.fit(train_data=train_data, test_data=test_data, epochs=epoch, batch_size=batch_size)


if __name__ == '__main__':
    print("################################ Prepare Data ############################")
    # TODO: change the data directory to [your data directory]
    data_dir = "../../data/modelnet40v1png/"
    train_dataset = MultiViewDataset6Party(data_dir, 'train', 32, 32, 6)
    valid_dataset = MultiViewDataset6Party(data_dir, 'test', 32, 32, 6)
    # class_lbls = ['person', 'animal']
    # train, test = load_two_party_data(data_dir, class_lbls, neg_label=0)
    # Xa_train, Xb_train, y_train = train
    # Xa_test, Xb_test, y_test = test

    batch_size = 32
    epoch = 50
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    run_experiment(train_data=train_loader, test_data=valid_loader, batch_size=args.batch_size, epoch=args.epochs)
