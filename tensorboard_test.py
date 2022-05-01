import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

w1 = SummaryWriter()
w2 = SummaryWriter()


for i in range(10000):
    w1.add_scalar("w1_test", i, global_step=i)
    w2.add_scalar("w2_test", -i, global_step = i)

while True:
    pass