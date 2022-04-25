import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(Discriminator, self).__init__()

        layers = [input_dim] + hidden_layers + [1]
        module = []
        for i in range(len(layers)-1):
            module.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-1:
                module.append(nn.ReLU())
        self.fc = nn.Sequential(*module)

    def forward(self, x):
        x = self.fc(x)
        return x

    def get_logits(self, x):
        return self.fc(x)