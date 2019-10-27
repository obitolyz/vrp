# -*- coding: utf-8 -*-
# @Time    : 2019/10/23 20:40
# @Author  : obitolyz
# @FileName: Struct2Vec.py
# @Software: PyCharm

import torch
import torch.nn as nn


class Struct2Vec(nn.Module):
    def __init__(self, service_num=20, p_dim=128, R=4):
        super(Struct2Vec, self).__init__()
        self.service_num = service_num
        self.p_dim = p_dim
        self.R = R
        self.theta_1 = nn.Linear(self.p_dim, self.p_dim, bias=False)  # mu
        self.theta_2 = nn.Linear(self.p_dim, self.p_dim, bias=False)  # ll-w
        self.theta_3 = nn.Linear(1, self.p_dim, bias=False)  # l-w

        self.theta_4 = nn.Linear(6, self.p_dim, bias=False)  # service node
        self.theta_5 = nn.Linear(2, self.p_dim, bias=False)  # depot node

    def forward(self, inputs):
        """
        :param inputs: [N x batch_size x 6]
        :return:
        """
        batch_size = inputs.size(1)
        N = self.service_num + 1
        mu = torch.zeros(N, batch_size, self.p_dim)
        mu_null = torch.zeros(N, batch_size, self.p_dim)
        for _ in range(self.R):
            for i in range(N):
                item_1 = self.theta_1(torch.sum(mu, dim=0) - mu[i])
                item_2 = self.theta_2(sum(
                    [torch.relu(self.theta_3(torch.norm(inputs[i][:, :2] - inputs[j][:, :2], dim=1, keepdim=True))) for
                     j in range(N)]))
                item_3 = self.theta_5(inputs[i][:, :2]) if i == 0 else self.theta_4(inputs[i])
                mu_null[i] = torch.relu(item_1 + item_2 + item_3)
            mu = mu_null.clone()

        return mu


if __name__ == "__main__":
    import numpy as np
    from torch.utils.data import DataLoader
    from Data_Generator import VRPDataset
    np.set_printoptions(suppress=True, threshold=np.inf)

    training_dataset = VRPDataset(service_num=20, num_samples=100)
    training_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=False, num_workers=1)
    model = Struct2Vec(service_num=20)
    for batch_id, sample_batch in enumerate(training_dataloader):
        # print(sample_batch.permute(1, 0, 2))
        mu = model(sample_batch.permute(1, 0, 2))
        print(mu.detach().numpy())
        break
    # print(mu.detach().numpy())
