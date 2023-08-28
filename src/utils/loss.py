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


class NotToRecommendLoss(nn.Module):
    """Improved BPRLoss with negative user feedback
    Learning from Negative User Feedback and Measuring Responsiveness for Sequential Recommenders
    RecSys ’23
            Args:
                - gamma(float): Small value to avoid division by zero

            Shape:
                - Pos_score: (N)
                - Neg_score: (N), same shape as the Pos_score
                - Output: scalar.

            Examples::

                >>> loss = ImprovedBPRLoss()
                >>> pos_score = torch.randn(3, requires_grad=True)
                >>> neg_score = torch.randn(3, requires_grad=True)
                >>> output = loss(pos_score, neg_score)
                >>> output.backward()
            """

    def __init__(self, gamma=1e-10):
        super(NotToRecommendLoss, self).__init__()
        self.gamma = gamma
        self.pos_weight = nn.Parameter(torch.tensor(1.0))
        self.neg_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, pos_score, neg_score):
        pos_loss = -torch.log(self.gamma + torch.sigmoid(pos_score)).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + self.gamma).mean()
        loss = (self.pos_weight * pos_loss + self.neg_weight * neg_loss) / (self.pos_weight + self.neg_weight)
        return loss


class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss
