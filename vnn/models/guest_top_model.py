import torch
import torch.nn as nn
import torch.optim as optim

from utils import get_logger

LOGGER = get_logger()


class GuestTopModelLearner(object):
    def __init__(self,
                 top_model,
                 classifier_criterion,
                 optim_dict,
                 logit_activation_fn=nn.Sigmoid()):
        self.top_model = top_model
        self.classifier_criterion = classifier_criterion
        self.logit_activation_fn = logit_activation_fn

        lr = optim_dict['learning_rate']
        mt = optim_dict['momentum']
        wd = optim_dict['weight_decay']
        self.optimizer = optim.SGD(self.top_model.parameters(), momentum=mt, weight_decay=wd, lr=lr)

    def forward(self, x):
        return self.top_model.forward(x)

    def predict(self, z_logit):
        z_logit_tensor = torch.tensor(z_logit, dtype=torch.float)
        output = self.top_model.forward(z_logit_tensor)
        return torch.sigmoid(output).detach().numpy()

    def train_top(self, z_logit, y):
        z_logit_tensor = torch.tensor(z_logit, requires_grad=True, dtype=torch.float)
        y_tensor = torch.tensor(y)

        activation = self.logit_activation_fn(z_logit_tensor) if self.logit_activation_fn else z_logit_tensor
        prediction = self.forward(activation)

        label_tensor = y_tensor.reshape(-1, 1).type_as(prediction)
        loss = self.classifier_criterion(prediction, label_tensor)

        loss.backward()
        backward_gradient = z_logit_tensor.grad.numpy()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return backward_gradient, loss.detach().numpy()


class GuestTopModel(nn.Module):
    def __init__(self, input_dim):
        super(GuestTopModel, self).__init__()
        self.classifier = nn.Sequential(
            # nn.Linear(in_features=input_dim, out_features=30),
            # nn.BatchNorm1d(30),
            # nn.LeakyReLU(),
            nn.Linear(in_features=input_dim, out_features=1)
        )

    def forward(self, x):
        return self.classifier(x)
