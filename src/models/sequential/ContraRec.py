#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：ContraRec.py
@Author     ：Heywecome
@Date       ：2023/9/21 14:40 
@Description：Sequential Recommendation with Multiple Contrast Signals TOIS'2022
python main.py --model_name ContraRec --emb_size 64 --lr 5e-4 --sample LS --ccc_temp 0.3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.BaseModel import SequentialModel
from models.init import xavier_uniform_initialization
from utils import layers
from utils.loss import BPRLoss


class ContraRec(SequentialModel):
    r"""
    In this version, we make \tau_{1} = 1 and K = 1 to ensure fair.
    Therefore, the ctc_loss == bpr_loss.
    In addition, we only implement the BERT4Rec as encoder, which is over-performed in most condition.
    """
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['gamma', 'num_neg', 'batch_size', 'ccc_temp']

    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--gamma', type=float, default=1,
                            help='Coefficient of the contrastive loss.')
        parser.add_argument('--beta_a', type=int, default=3,
                            help='Parameter of the beta distribution for sampling.')
        parser.add_argument('--beta_b', type=int, default=3,
                            help='Parameter of the beta distribution for sampling.')
        parser.add_argument('--ccc_temp', type=float, default=0.2,
                            help='Temperature in context-context contrastive loss.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.max_his = args.history_max
        self.gamma = args.gamma
        self.beta_a = args.beta_a
        self.beta_b = args.beta_b
        self.ccc_temp = args.ccc_temp
        self.mask_token = corpus.n_items
        self.item_embeddings = nn.Embedding(self.item_num + 1, self.emb_size)
        self.encoder = BERT4RecEncoder(self.emb_size, self.max_his, num_layers=2, num_heads=2)
        self.ccc_loss = ContraLoss(self.device, temperature=self.ccc_temp)
        self.apply(xavier_uniform_initialization)

    def forward(self, feed_dict: dict):
        history = feed_dict['history_items']  # bsz, history_max
        lengths = feed_dict['lengths']  # bsz

        his_vectors = self.item_embeddings(history)
        his_vector = self.encoder(his_vectors, lengths)

        return his_vector


    def predict(self, feed_dict):
        pos_item = feed_dict['pos_item']  # [batch_size, -1]
        neg_item = feed_dict['neg_items']
        lengths = feed_dict['lengths']  # bsz

        pos_e = self.item_embeddings(pos_item)
        neg_e = self.item_embeddings(neg_item)
        user_e = self.forward(feed_dict)

        if feed_dict['phase'] == 'train':
            history_a = feed_dict['history_items_a']
            his_a_vectors = self.item_embeddings(history_a)
            his_a_vector = self.encoder(his_a_vectors, lengths)
            history_b = feed_dict['history_items_b']
            his_b_vectors = self.item_embeddings(history_b)
            his_b_vector = self.encoder(his_b_vectors, lengths)
            features = torch.stack([his_a_vector, his_b_vector], dim=1)  # bsz, 2, emb
            features = F.normalize(features, dim=-1) # bsz, 2, emb
            labels = pos_item  # bsz

        pos_item_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_item_score = torch.mul(user_e, neg_e).sum(dim=1)
        bpr_loss = BPRLoss()
        ctc_loss = bpr_loss(pos_item_score, neg_item_score)

        ccc_loss = self.ccc_loss(features=features, labels=labels)
        loss = ctc_loss + self.gamma * ccc_loss

        return loss

    def full_predict(self, feed_dict):
        user_e = self.forward(feed_dict)
        item_e = self.item_embeddings.weight

        # expand the user embedding to match the shape of items
        scores = torch.matmul(user_e, item_e.transpose(0, 1))  # (batch_size, item_num)
        return scores

    class Dataset(SequentialModel.Dataset):
        def reorder_op(self, seq):
            ratio = np.random.beta(a=self.model.beta_a, b=self.model.beta_b)
            select_len = int(len(seq) * ratio)
            start = np.random.randint(0, len(seq) - select_len + 1)
            idx_range = np.arange(len(seq))
            np.random.shuffle(idx_range[start: start + select_len])
            return seq[idx_range]

        def mask_op(self, seq):
            ratio = np.random.beta(a=self.model.beta_a, b=self.model.beta_b)
            selected_len = int(len(seq) * ratio)
            mask = np.full(len(seq), False)
            mask[:selected_len] = True
            np.random.shuffle(mask)
            seq[mask] = self.model.mask_token
            return seq

        def augment(self, seq):
            aug_seq = np.array(seq).copy()
            if np.random.rand() > 0.5:
                return self.mask_op(aug_seq)
            else:
                return self.reorder_op(aug_seq)

        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            if self.phase == 'train':
                history_items_a = self.augment(feed_dict['history_items'])
                history_items_b = self.augment(feed_dict['history_items'])
                feed_dict['history_items_a'] = history_items_a
                feed_dict['history_items_b'] = history_items_b
            return feed_dict


class ContraLoss(nn.Module):
    def __init__(self, device, temperature=0.2):
        super(ContraLoss, self).__init__()
        # Store provided device and temperature parameters
        self.device = device
        self.temperature = temperature

    def forward(self, features, labels=None):
        """
        Here, the forward function calculates the contrastive loss through several steps.
        Args:
            features: hidden vector of shape [bsz, n_views, dim].
            labels: target item of shape [bsz].
        Returns:
            A loss scalar.
        """
        # Get the batch size from the features
        batch_size = features.shape[0]

        # Obtain the label mask. If no labels are provided, create an identity matrix.
        # Otherwise, compare each label pair and create a binary matrix of equal labels.
        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        else:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.transpose(0, 1)).float().to(self.device)

        # The contrast count is the number of views
        contrast_count = features.shape[1]

        # The contrast feature is obtained by concatenating all features along the dimension 0
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # Compute the similarity matrix (logits), divide each entry by temperature to soften probabilities
        anchor_dot_contrast = torch.matmul(contrast_feature, contrast_feature.transpose(0, 1)) / self.temperature
        # We subtract the maximum value for stability during the exponential operation in softmax computation.
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # Using .sub_() for inplace operation to reduce memory footprint and speed up computation
        anchor_dot_contrast.sub_(logits_max)
        logits = anchor_dot_contrast - logits_max.detach()

        # Expand the mask matrix to match the dimension of the contrast matrix
        mask = mask.repeat(contrast_count, contrast_count)

        # Generate a mask to avoid a sample to be contrasted with itself
        # The scatter_ operation puts "0" in each diagonal entry
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * contrast_count).view(-1, 1).to(self.device), 0
        )
        mask = mask * logits_mask

        # Compute log probability for each pair; apply the mask to exclude invalid samples
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        # Compute the mean of log-likelihood over positive (similar pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)

        # The final loss is the negative of the temperature-scaled mean log probability
        loss = -self.temperature * mean_log_prob_pos
        return loss.mean()  # Return the mean loss over all samples in the batch


class BERT4RecEncoder(nn.Module):
    def __init__(self, emb_size, max_his, num_layers=2, num_heads=2):
        super().__init__()
        self.p_embeddings = nn.Embedding(max_his + 1, emb_size)
        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=emb_size, d_ff=emb_size, n_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, seq, lengths):
        batch_size, seq_len = seq.size(0), seq.size(1)
        len_range = torch.from_numpy(np.arange(seq_len)).to(seq.device)
        valid_mask = len_range[None, :] < lengths[:, None]

        # Position embedding
        position = len_range[None, :] * valid_mask.long()
        pos_vectors = self.p_embeddings(position)
        seq = seq + pos_vectors

        # Self-attention
        attn_mask = valid_mask.view(batch_size, 1, 1, seq_len)
        for block in self.transformer_block:
            seq = block(seq, attn_mask)
        seq = seq * valid_mask[:, :, None].float()

        his_vector = seq[torch.arange(batch_size), lengths - 1]
        return his_vector
