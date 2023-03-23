#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：POP.py
@Author  ：Heywecome
@Date    ：2023/3/23 09:19
'''
import numpy as np
import torch

from models.BaseModel import GeneralModel


class POP(GeneralModel):

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.popularity = np.zeros(corpus.n_items)
        for i in corpus.data_df['train']['item_id'].values:
            self.popularity[i] += 1

    def forward(self, feed_dict: dict) -> dict:
        self.check_list = []
        i_ids = feed_dict['item_id']
        prediction = self.popularity[i_ids.cpu().data.numpy()]
        prediction = torch.from_numpy(prediction).to(self.device)
        return {'prediction': prediction}
