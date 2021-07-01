import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class DenseModel(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate=0.01, bias=True):
        super(DenseModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim, bias=bias),
        )
        self.is_debug = False
        self.optimizer = optim.SGD(self.parameters(), momentum=0.99, weight_decay=0.01, lr=learning_rate)

    def forward(self, x):
        if self.is_debug: print("[DEBUG] DenseModel.forward")

        x = torch.tensor(x).float()
        return self.classifier(x).detach().numpy()

    def backward(self, x, grads):
        if self.is_debug: print("[DEBUG] DenseModel.backward")

        x = torch.tensor(x, requires_grad=True).float()
        grads = torch.tensor(grads).float()
        output = self.classifier(x)
        output.backward(gradient=grads)
        x_grad = x.grad.numpy()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return x_grad


class LocalModel(nn.Module):
    def __init__(self, input_dim, output_dim, optimizer_dict):
        super(LocalModel, self).__init__()
        # self.extractor = nn.Sequential(
        #     nn.Linear(in_features=input_dim, out_features=output_dim),
        #     nn.LeakyReLU()
        # )
        self.extractor = models.resnet18(pretrained=False, num_classes=output_dim)
        # self.extractor.fc = nn.Linear(512, output_dim)
        self.output_dim = output_dim
        self.is_debug = False

        lr = optimizer_dict['learning_rate']
        momentum = optimizer_dict['momentum']
        weight_decay = optimizer_dict['weight_decay']
        self.optimizer = optim.SGD(self.parameters(), momentum=momentum, weight_decay=weight_decay, lr=lr)

    def forward(self, x):
        if self.is_debug: print("[DEBUG] DenseModel.forward")

        # x = torch.tensor(x)
        return self.extractor(x).detach().numpy()

    def predict(self, x):
        if self.is_debug: print("[DEBUG] DenseModel.predict")

        x = torch.tensor(x).float()
        return self.extractor(x).detach().numpy()

    def backward(self, x, grads):
        if self.is_debug: print("[DEBUG] DenseModel.backward")

        # x = torch.tensor(x).float()
        grads = torch.tensor(grads).float()
        output = self.extractor(x)
        output.backward(gradient=grads)

        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_output_dim(self):
        return self.output_dim
