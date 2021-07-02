import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from vlr.vfl import VerticalMultiplePartyLogisticRegressionFederatedLearning


def compute_correct_prediction(*, y_targets, y_prob_preds, threshold=0.5):
    y_targets = y_targets.squeeze(1)
    y_prob_preds = torch.tensor(y_prob_preds)
    y_preds = torch.argmax(y_prob_preds, dim=1)
    num_correct = torch.eq(y_preds, y_targets).sum().float().item()
    counts = len(y_targets)
    # y_hat_lbls = []
    # pred_pos_count = 0
    # pred_neg_count = 0
    # correct_count = 0
    # for y_prob, y_t in zip(y_prob_preds, y_targets):
    #     if y_prob <= threshold:
    #         pred_neg_count += 1
    #         y_hat_lbl = 0
    #     else:
    #         pred_pos_count += 1
    #         y_hat_lbl = 1
    #     y_hat_lbls.append(y_hat_lbl)
    #     if y_hat_lbl == y_t:
    #         correct_count += 1

    return num_correct, counts


class FederatedLearningFixture(object):

    def __init__(self, federated_learning):
        self.federated_learning = federated_learning

    def fit(self, train_data, test_data, epochs=50, batch_size=-1):

        # TODO: add early stopping
        main_party_id = self.federated_learning.get_main_party_id()
        print(main_party_id)
        train_loader = train_data[main_party_id]["X"]
        # y_train = train_data[main_party_id]["Y"]
        test_loader = test_data[main_party_id]["X"]
        # y_test = test_data[main_party_id]["Y"]

        # N = Xa_train.shape[0]
        # residual = N % batch_size
        # if residual == 0:
        #     n_batches = N // batch_size
        # else:
        #     n_batches = N // batch_size + 1

        print("number of samples:", len(train_loader) * batch_size)
        print("batch size:", batch_size)
        print("number of batches:", len(train_loader))

        global_step = -1
        recording_period = 30
        recording_step = -1
        threshold = 0.5

        loss_list = []
        running_time_list = []
        for ep in range(epochs):
            # for batch_idx in range(n_batches):
            self.federated_learning.guest_local_model.train()
            self.federated_learning.guest_intr_layer.guest_dense_model.internal_dense_model.train()
            self.federated_learning.guest_intr_layer.host_dense_model_dict['A'].internal_dense_model.train()
            self.federated_learning.guest_top_learner.top_model.train()
            self.federated_learning.host_local_model_dict['A'].train()
            for i, (X, y) in enumerate(tqdm(train_loader)):
                global_step += 1

                # prepare batch data for party A, which has both X and y.
                Xa_batch = X[0]
                Y_batch = y
                # Xb_batch = X[1]
                # prepare batch data for all other parties, which only has both X.
                party_X_train_batch_dict = dict()
                for idx, party_id in enumerate(train_data["party_list"].items()):
                    party_X_train_batch_dict[party_id[0]] = X[1:][idx]

                loss = self.federated_learning.fit(Xa_batch, Y_batch,
                                                   party_X_train_batch_dict,
                                                   global_step)
                loss_list.append(loss)
                if (global_step + 1) % recording_period == 0:
                    recording_step += 1
                    avg_loss = np.mean(loss_list)
                    print("===> epoch: {0}, batch: {1}, loss: {2}"
                          .format(ep, i, avg_loss))
            total_corrects = 0
            total_counts = 0
            self.federated_learning.guest_local_model.eval()
            self.federated_learning.guest_intr_layer.guest_dense_model.internal_dense_model.eval()
            self.federated_learning.guest_intr_layer.host_dense_model_dict['A'].internal_dense_model.eval()
            self.federated_learning.guest_top_learner.top_model.eval()
            self.federated_learning.host_local_model_dict['A'].eval()
            with torch.no_grad():
                for i, (X, y) in enumerate(tqdm(test_loader)):
                    party_X_test_dict = dict()
                    for idx, party_id in enumerate(test_data["party_list"].items()):
                        party_X_test_dict[party_id[0]] = X[1:][idx]
                    y_prob_preds = self.federated_learning.predict(X[0], party_X_test_dict)
                    num_correct, counts = compute_correct_prediction(y_targets=y, y_prob_preds=y_prob_preds,
                                                                     threshold=threshold)
                    total_corrects += num_correct
                    total_counts += counts
                acc = total_corrects / total_counts
                print("===> epoch: {0}, acc: {1}".format(ep, acc))
            # acc = accuracy_score(y_test, y_hat_lbls)
            # auc = roc_auc_score(y_test, y_prob_preds)
            # print("===> epoch: {0}, batch: {1}, loss: {2}, acc: {3}, auc: {4}"
            #       .format(ep, batch_idx, ave_loss, acc, auc))
            # print("===>", precision_recall_fscore_support(y_test, y_hat_lbls, average="macro"))
            # loss_list = list()
            # party_X_test_dict = dict()
            # for party_id, party_X in test_data["party_list"].items():
            #     party_X_test_dict[party_id] = party_X
            # y_prob_preds = self.federated_learning.predict(Xa_test, party_X_test_dict)
            # y_hat_lbls, statistics = compute_correct_prediction(y_targets=y_test,
            #                                                     y_prob_preds=y_prob_preds,
            #                                                     threshold=threshold)
            # acc = accuracy_score(y_test, y_hat_lbls)
            # auc = roc_auc_score(y_test, y_prob_preds)
            # print("===> epoch: {0}, batch: {1}, loss: {2}, acc: {3}, auc: {4}"
            #       .format(ep, batch_idx, ave_loss, acc, auc))
            # print("===>", precision_recall_fscore_support(y_test, y_hat_lbls, average="macro"))
