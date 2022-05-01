import numpy as np
import copy
import torch
from learning.normalizer import Normalizer

class TFNormalizer(Normalizer):

    def __init__(self, size, groups_ids=None, eps=0.02, clip=np.inf):
        super().__init__(size, groups_ids, eps, clip)

        self._build_resource_tf()
        return

    # initialze count when loading saved values so that things don't change to quickly during updates
    def load(self):
        self.count = self.count_tf.numpy()[0]
        self.mean = self.mean_tf.numpy()
        self.std = self.std_tf.numpy()
        self.mean_sq = self.calc_mean_sq(self.mean, self.std)
        return

    def update(self):
        super().update()
        self._update_resource_tf()
        return

    def set_mean_std(self, mean, std):
        super().set_mean_std(mean, std)
        self._update_resource_tf()
        return

    def normalize_tf(self, x):
        x = torch.FloatTensor(x)
        norm_x = (x - self.mean_tf) / self.std_tf
        norm_x = torch.clip(norm_x, -self.clip, self.clip)
        return norm_x

    def unnormalize_tf(self, norm_x):
        x = norm_x * self.std_tf + self.mean_tf
        return x
    
    def _build_resource_tf(self):
        with torch.no_grad():
            self.count_tf = torch.Tensor(np.array([self.count], dtype=np.int32))
            self.mean_tf = torch.FloatTensor(self.mean.astype(np.float32))
            self.std_tf = torch.FloatTensor(self.std.astype(np.float32))
        
        return

    def _update_resource_tf(self):
        with torch.no_grad():
            self.count_tf = torch.Tensor(np.array([self.count], dtype=np.int32))
            self.mean_tf = torch.FloatTensor(self.mean.astype(np.float32))
            self.std_tf = torch.FloatTensor(self.std.astype(np.float32))
        return
