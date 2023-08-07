#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：loss.py
@Author     ：Heywecome
@Date       ：2023/8/4 14:11 
@Description：Common loss in recommender system
"""
import torch
import torch.nn as nn

class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

        Args:
            - gamma(float): Small value to avoid division by zero

        Shape:
            - Pos_score: (N)
            - Neg_score: (N), same shape as the Pos_score
            - Output: scalar.

        Examples::

            >>> loss = BPRLoss()
            >>> pos_score = torch.randn(3, requires_grad=True)
            >>> neg_score = torch.randn(3, requires_grad=True)
            >>> output = loss(pos_score, neg_score)
            >>> output.backward()
        """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss