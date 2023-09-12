import math
import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np

from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List
from utils import utils
from models.BaseModel import BaseModel


class BaseRunner(object):
    @staticmethod
    def parse_runner_args(parser, configs):
        parser.add_argument('--epoch', type=int, default=200,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check some tensors every check_epoch.')
        parser.add_argument('--test_epoch', type=int, default=-1,
                            help='Print test results every test_epoch (-1 means no print).')
        parser.add_argument('--early_stop', type=int, default=10,
                            help='The number of epochs when dev results drop continuously.')
        parser.add_argument('--lr', type=float, default=5e-4,
                            help='Learning rate.')
        parser.add_argument('--l2', type=float, default=0,
                            help='Weight decay in optimizer.')
        parser.add_argument('--batch_size', type=int, default=256,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=256,
                            help='Batch size during testing.')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: SGD, Adam, Adagrad, Adadelta')
        parser.add_argument('--num_workers', type=int, default=5,
                            help='Number of processors when prepare batches in DataLoader')
        parser.add_argument('--pin_memory', type=int, default=0,
                            help='pin_memory in DataLoader')
        parser.add_argument('--topk', type=str, default='10,20',
                            help='The number of items recommended to each user.')
        parser.add_argument('--metric', type=str, default='NDCG, HR',
                            help='metrics: NDCG, RECALL')

        args, extras = parser.parse_known_args()

        # Update the configs dictionary with the parsed arguments
        configs['runner']['epoch'] = args.epoch
        configs['runner']['check_epoch'] = args.check_epoch
        configs['runner']['test_epoch'] = args.test_epoch
        configs['runner']['early_stop'] = args.early_stop
        configs['runner']['lr'] = args.lr
        configs['runner']['l2'] = args.l2
        configs['runner']['batch_size'] = args.batch_size
        configs['runner']['eval_batch_size'] = args.eval_batch_size
        configs['runner']['optimizer'] = args.optimizer
        configs['runner']['num_workers'] = args.num_workers
        configs['runner']['pin_memory'] = args.pin_memory
        configs['runner']['topk'] = args.topk
        configs['runner']['metric'] = args.metric

        return parser

    @staticmethod
    def evaluate_method(predictions: np.ndarray, topk: list, metrics: list) -> Dict[str, float]:
        """
        :param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
        :param topk: top-K value list
        :param metrics: metric string list
        :return: a result dict, the keys are metric@topk
        To compute the NDCG (Normalized Discounted Cumulative Gain) and Recall values,
        we can iterate over the relevance thresholds in the same manner as we did for the hit-rate.
        For NDCG@K, it's computed as:
        ndcg = (1 / log2(rank + 1)) * hit
        This is assuming you only have binary relevance (either an item is relevant or it isn't
        - as is the case with implicit feedback).

        The NDCG is then simply the mean of these values.
        For recall@K, you need to divide the number of hits at rank K by the total number of relevant items.
        Since we're dealing with implicit feedback where there's only one relevant item per row,
        this is equivalent to hit@K:
        recall = hit.
        """
        evaluations = dict()

        # Sort predictions and find ranks
        sort_idx = (-predictions).argsort(axis=1)
        gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1

        # Create a mask of hits within topk ranks for each value in topk
        hits = np.asarray([gt_rank <= k for k in topk])
        for i, k in enumerate(topk):
            hit = hits[i]
            for metric in metrics:
                key = f'{metric}@{k}'
                if metric.lower() == 'hr':
                    evaluations[key] = hit.mean()
                elif metric.lower() == 'ndcg':
                    denominator = np.log2(np.clip(gt_rank, 1, k) + 1)
                    evaluations[key] = np.mean(hit / denominator)
                elif metric.lower() == 'recall' or metric.lower() == 'mrr':
                    evaluations[key] = hit.mean()
                elif metric.lower() == 'precision':
                    evaluations[key] = hit.sum() / (hit.shape[0] * k)  # hit.shape[0] is equivalent to len(hit)
                else:
                    raise ValueError(f'Undefined evaluation metric: {metric}.')

        return evaluations

    def __init__(self, configs):
        self.epoch = configs['runner']['epoch']
        self.check_epoch = configs['runner']['check_epoch']
        self.test_epoch = configs['runner']['test_epoch']
        self.early_stop = configs['runner']['early_stop']
        self.learning_rate = float(configs['runner']['lr'])
        self.batch_size = configs['runner']['batch_size']
        self.eval_batch_size = configs['runner']['eval_batch_size']
        self.l2 = configs['runner']['l2']
        self.optimizer_name = configs['runner']['optimizer']
        self.num_workers = configs['runner']['num_workers']
        self.pin_memory = configs['runner']['pin_memory']
        self.topk = [int(x) for x in configs['runner']['topk'].split(',')]
        self.metrics = [m.strip().upper() for m in configs['runner']['metric'].split(',')]
        self.main_metric = '{}@{}'.format(self.metrics[0], self.topk[0])  # early stop based on main_metric

        self.time = None  # will store [start_time, last_step_time]

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def _build_optimizer(self, model):
        logging.info('Optimizer: ' + self.optimizer_name)
        optimizer = eval('torch.optim.{}'.format(self.optimizer_name))(
            model.parameters(), lr=self.learning_rate, weight_decay=self.l2)
        return optimizer

    def train(self, data_dict: Dict[str, BaseModel.Dataset]):
        model = data_dict['train'].model
        main_metric_results, dev_results = list(), list()
        self._check_time(start=True)
        try:
            for epoch in range(self.epoch):
                # Fit
                self._check_time()
                gc.collect()
                torch.cuda.empty_cache()
                loss = self.fit(data_dict['train'], epoch=epoch + 1)
                training_time = self._check_time()

                # Observe selected tensors
                if len(model.check_list) > 0 and self.check_epoch > 0 and epoch % self.check_epoch == 0:
                    utils.check(model.check_list)

                # Record dev results
                dev_result = self.evaluate(data_dict['dev'], self.topk[:1], self.metrics)
                dev_results.append(dev_result)
                main_metric_results.append(dev_result[self.main_metric])
                logging_str = 'Epoch {:<5} loss={:<.4f} [{:<3.1f} s]    dev=({})'.format(
                    epoch + 1, loss, training_time, utils.format_metric(dev_result))

                # Test
                if self.test_epoch > 0 and epoch % self.test_epoch == 0:
                    test_result = self.evaluate(data_dict['test'], self.topk[:1], self.metrics)
                    logging_str += ' test=({})'.format(utils.format_metric(test_result))
                testing_time = self._check_time()
                logging_str += ' [{:<.1f} s]'.format(testing_time)

                # Save model and early stop
                if max(main_metric_results) == main_metric_results[-1] or \
                        (hasattr(model, 'stage') and model.stage == 1):
                    model.save_model()
                    logging_str += ' *'
                logging.info(logging_str)

                if self.early_stop > 0 and self.eval_termination(main_metric_results):
                    logging.info("Early stop at %d based on dev result." % (epoch + 1))
                    break
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith('y'):
                logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                exit(1)

        # Find the best dev result across iterations
        best_epoch = main_metric_results.index(max(main_metric_results))
        logging.info(os.linesep + "Best Iter(dev)={:>5}\t dev=({}) [{:<.1f} s] ".format(
            best_epoch + 1, utils.format_metric(dev_results[best_epoch]), self.time[1] - self.time[0]))
        model.load_model()

    def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
        model = dataset.model
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        dataset.actions_before_epoch()  # must sample before multi thread start

        model.train()
        loss_lst = list()
        dl = DataLoader(dataset,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.num_workers,
                        collate_fn=dataset.collate_batch,
                        pin_memory=self.pin_memory)
        for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
            batch = utils.batch_to_gpu(batch, model.device)
            model.optimizer.zero_grad()
            loss = model.predict(batch)
            loss.backward()
            model.optimizer.step()
            loss_lst.append(loss.detach().cpu().data.numpy())
        return np.mean(loss_lst).item()

    def eval_termination(self, criterion: List[float]) -> bool:
        if len(criterion) > self.early_stop and utils.non_increasing(criterion[-self.early_stop:]):
            return True
        elif len(criterion) - criterion.index(max(criterion)) > self.early_stop:
            return True
        return False

    def evaluate(self, dataset: BaseModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
        """
        Evaluate the results for an input dataset.
        :return: result dict (key: metric@k)
        """
        predictions = self.interface(dataset)
        return self.evaluate_method(predictions, topks, metrics)

    def interface(self, dataset: BaseModel.Dataset) -> np.ndarray:
        """
        The returned prediction is a 2D-array, each row corresponds to all the candidates,
        and the ground-truth item poses the first.
        Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
                 predictions like: [[1,3,4], [2,5,6]]
        """
        model = dataset.model
        model.eval()
        predictions, target_scores_list, scores_list = [], [], []

        dl = DataLoader(dataset,
                        batch_size=self.eval_batch_size,
                        shuffle=False,
                        num_workers=self.num_workers,
                        collate_fn=dataset.collate_batch,
                        pin_memory=self.pin_memory)
        for batch in tqdm(dl, leave=False, desc='Predict:', ncols=100, mininterval=1):
            batch = utils.batch_to_gpu(batch, model.device)
            scores = model.full_predict(batch)
            target_scores = scores[torch.arange(len(batch['pos_item'])), batch['pos_item']]
            target_scores_list.append(target_scores)
            scores_list.append(scores)

        target_scores = torch.cat(target_scores_list, dim=0).detach()
        scores = torch.cat(scores_list, dim=0).detach()

        # Precompute clicked item unions
        clicked_item_unions = {
            user_id: dataset.corpus.train_clicked_set[user_id].union(dataset.corpus.residual_clicked_set[user_id])
            for user_id in dataset.data['user_id']
        }

        # mask the score of items that users have interacted with, to -np.inf
        if dataset.model.test_all:
            for user_idx, user_id in enumerate(dataset.data['user_id']):
                clicked_items = clicked_item_unions[user_id]
                scores[user_idx, list(clicked_items)] = -np.inf

        predictions = torch.cat([target_scores.unsqueeze(1), scores], dim=1).cpu().numpy()
        return predictions

    def print_res(self, dataset: BaseModel.Dataset) -> str:
        """
        Construct the final result string before/after training
        :return: test result string
        """
        result_dict = self.evaluate(dataset, self.topk, self.metrics)
        res_str = '(' + utils.format_metric(result_dict) + ')'
        return res_str
