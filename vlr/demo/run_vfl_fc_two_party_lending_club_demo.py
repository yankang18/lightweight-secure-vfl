from sklearn.utils import shuffle

from datasets.lending_club_dataset import load_two_party_data
from vlr.vfl_fixture import FederatedLearningFixture
from vlr.party_models import VFLGuestModel, VFLHostModel
from vlr.pytorch_models import LocalModel, DenseModel
from vlr.vfl import VerticalMultiplePartyLogisticRegressionFederatedLearning


def run_experiment(train_data, test_data, batch_size, learning_rate, epoch):
    print("hyper-parameters:")
    print("batch size: {0}".format(batch_size))
    print("learning rate: {0}".format(learning_rate))

    Xa_train, Xb_train, y_train = train_data
    Xa_test, Xb_test, y_test = test_data

    print("################################ Wire Federated Models ############################")

    # create local models for both party A and party B
    party_a_local_model = LocalModel(input_dim=Xa_train.shape[1], output_dim=10, learning_rate=learning_rate)
    party_b_local_model = LocalModel(input_dim=Xb_train.shape[1], output_dim=20, learning_rate=learning_rate)

    # create lr model for both party A and party B. Each party has a part of the whole lr model and only party A has
    # the bias since only party A has the labels.
    party_a_dense_model = DenseModel(party_a_local_model.get_output_dim(), 1, learning_rate=learning_rate, bias=True)
    party_b_dense_model = DenseModel(party_b_local_model.get_output_dim(), 1, learning_rate=learning_rate, bias=False)
    partyA = VFLGuestModel(local_model=party_a_local_model)
    partyA.set_dense_model(party_a_dense_model)
    partyB = VFLHostModel(local_model=party_b_local_model)
    partyB.set_dense_model(party_b_dense_model)

    party_B_id = "B"
    federatedLearning = VerticalMultiplePartyLogisticRegressionFederatedLearning(partyA)
    federatedLearning.add_party(id=party_B_id, party_model=partyB)
    federatedLearning.set_debug(is_debug=False)

    print("################################ Train Federated Models ############################")

    fl_fixture = FederatedLearningFixture(federatedLearning)

    # only party A has labels (i.e., Y), other parties only have features (e.g., X).
    # 'party_list' stores X for all other parties.
    # Since this is two-party VFL, 'party_list' only stores the X of party B.
    train_data = {federatedLearning.get_main_party_id(): {"X": Xa_train, "Y": y_train},
                  "party_list": {party_B_id: Xb_train}}
    test_data = {federatedLearning.get_main_party_id(): {"X": Xa_test, "Y": y_test},
                 "party_list": {party_B_id: Xb_test}}

    fl_fixture.fit(train_data=train_data, test_data=test_data, epochs=epoch, batch_size=batch_size)


if __name__ == '__main__':
    print("################################ Prepare Data ############################")
    data_dir = "../../../../../Data/lending_club_bundle_archive/"
    train, test = load_two_party_data(data_dir)
    Xa_train, Xb_train, y_train = train
    Xa_test, Xb_test, y_test = test

    batch_size = 256
    epoch = 100
    lr = 0.01

    Xa_train, Xb_train, y_train = shuffle(Xa_train, Xb_train, y_train)
    Xa_test, Xb_test, y_test = shuffle(Xa_test, Xb_test, y_test)
    train = [Xa_train, Xb_train, y_train]
    test = [Xa_test, Xb_test, y_test]
    run_experiment(train_data=train, test_data=test, batch_size=batch_size, learning_rate=lr, epoch=epoch)
