#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：WindingSignal.py
@Author     ：Heywecome
@Date       ：2023/3/31 14:56
@Description：Winding Signal for recommendation
CMD example:
python main.py --model_name Sig4Rec --emb_size 64 --include_attr 1 --lr 1e-3 --history_max 20 --num_neg 1 --batch_size 4096 --dataset 'Food' --sample_nums 128
"""
import logging
import math
import random

import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

from utils import layers
from models.BaseModel import SequentialModel
from helpers.KGReader import KGReader
from numpy.fft import fftfreq

"""
Ready for submission.
We will release the code after accept.
"""