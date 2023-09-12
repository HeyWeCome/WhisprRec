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
    @staticmethod
    def parse_model_args(parser, configs):
        parser.add_argument('--embedding_size', type=int, default=64,
                            help='Size of embedding vectors.')
        args, extras = parser.parse_known_args()
        # Update the configs dictionary with the parsed arguments
        configs['model']['embedding_size'] = args.embedding_size
        return parser

    def __init__(self, corpus, configs):
        super().__init__(corpus, configs)

        # load parameter info
        self.emb_size = configs['model']['embedding_size']

        # define layers and loss
        self.user_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.item_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.loss = BPRLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

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

    def predict(self, feed_dict):
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

    def full_predict(self, feed_dict):
        user = feed_dict['user_id']

        user_e = self.get_user_embedding(user)
        item_e = self.item_embeddings.weight

        # expand the user embedding to match the shape of neg_items
        neg_scores = torch.matmul(user_e, item_e.transpose(0, 1))  # (batch_size, neg_item_num)

        return neg_scores
