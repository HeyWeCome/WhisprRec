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
        self.emb_size = args.emb_size
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

    def forward(self, feed_dict):
        self.check_list = []
        u_ids = feed_dict['user_id']  # [batch_size] [256]
        i_ids = feed_dict['item_id']  # [batch_size, -1] [256, 2]

        u_emb = self.u_embeddings(u_ids)  # [batch_size, emb_size] [256, 64]
        i_emb = self.i_embeddings(i_ids)  # [batch_size, 2, emb_size] [256, 2, 64]

        # training: prediction[batch_size, 2]
        # testing: prediction[batch_size, 100]
        # [256, 1, emb_size] * [256, 2, emb_size] -> [256, 2, emb_size] -> [256, 2]
        # In training phrase, first is positive, second is negative
        prediction = torch.sum(torch.mul(torch.unsqueeze(u_emb, 1), i_emb), dim=-1)
        return {'prediction': prediction}
