""" BPRMF
Reference:
    "Bayesian personalized ranking from implicit feedback"
    Rendle et al., UAI'2009.
CMD example:
    python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset 'ml-100k'
"""

import torch
import torch.nn as nn

from models.BaseModel import GeneralModel
from models.init import xavier_normal_initialization
from utils.loss import BPRLoss


class BPRMF(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)

        # load parameter info
        self.emb_size = args.emb_size

        # define layers and loss
        self.user_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.item_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.loss = BPRLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    # def forward(self, feed_dict):
    #     self.check_list = []
    #     u_ids = feed_dict['user_id']  # [batch_size] [256]
    #     pos_id = feed_dict['pos_item']
    #     neg_ids = feed_dict['neg_items']
    #
    #     # i_ids = feed_dict['item_id']  # [batch_size, -1] [256, 2]
    #
    #     u_emb = self.user_embeddings(u_ids)  # [batch_size, emb_size] [256, 64]
    #     pos_emb = self.item_embeddings(pos_id)
    #     neg_emb = self.item_embeddings(neg_ids)
    #
    #     u_emb_expanded = u_emb.unsqueeze(1).repeat(1, neg_emb.size(1), 1)
    #     pos_item_score = torch.mul(u_emb, pos_emb)
    #     neg_item_score = torch.mul(u_emb_expanded, neg_emb).sum(dim=-1)
    #     loss = self.loss(pos_item_score, pos_item_score)
    #     # i_emb = self.item_embeddings(i_ids)  # [batch_size, 2, emb_size] [256, 2, 64]
    #
    #     # prediction = torch.sum(torch.mul(torch.unsqueeze(u_emb, 1), i_emb), dim=-1)
    #     # return {'prediction': prediction}
    #     return pos_item_score, neg_item_score, loss

    def get_user_embedding(self, user):
        r"""Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.user_embeddings(user)

    def get_item_embedding(self, item):
        r"""Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embeddings(item)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, feed_dict):
        user = feed_dict['user_id']
        pos_item = feed_dict['pos_item']
        neg_item = feed_dict['neg_items']

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_item_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_item_score = torch.mul(user_e, neg_e).sum(dim=1)
        bpr_loss = BPRLoss()
        loss = bpr_loss(pos_item_score, neg_item_score)
        return loss

    def predict(self, feed_dict):
        user = feed_dict['user_id']
        pos_item = feed_dict['pos_item']
        neg_item = feed_dict['neg_items']

        user_e = self.get_user_embedding(user)
        pos_e = self.get_item_embedding(pos_item)
        neg_e = self.get_item_embedding(neg_item)

        pos_scores = (user_e * pos_e).sum(dim=-1)  # (batch_size,)
        # expand the user embedding to match the shape of neg_items
        user_e = user_e.unsqueeze(1)  # (batch_size, 1, emb_size)
        neg_scores = (user_e * neg_e).sum(dim=-1)  # (batch_size, neg_item_num)

        return pos_scores, neg_scores



