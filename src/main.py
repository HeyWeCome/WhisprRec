import os
import sys
import pickle
import logging
import argparse
import pandas as pd
import torch
import yaml

from helpers import *
from helpers.BaseRunner import BaseRunner
# from models.developing import *
# from models.graph import *
from models.general import *
from models.sequential import *
from utils import utils


def parse_global_args(parser):
    parser.add_argument('--gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES, default for CPU only')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file path')
    parser.add_argument('--random_seed', type=int, default=3407,
                        help='Random seed of numpy and pytorch, and 3407 is all you need')
    parser.add_argument('--load', type=int, default=0,
                        help='Whether load model and continue to train')
    parser.add_argument('--train', type=int, default=1,
                        help='To train the model or not.')
    parser.add_argument('--regenerate', type=int, default=1,
                        help='Whether to regenerate intermediate files')
    return parser


def main(configs):
    # Random seed
    # Fix the problem caused by numpy
    utils.init_seed(configs['random_seed'])

    # GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = configs['gpu']
    configs['device'] = torch.device('cpu')
    if configs['gpu'] != '' and torch.cuda.is_available():
        configs['device'] = torch.device('cuda')
    logging.info('Device: {}'.format(configs['device']))

    # Read data
    corpus_path = os.path.join(configs['reader']['path'],
                               configs['reader']['dataset'],
                               configs['reader']['name'] + '.pkl')
    if not configs['regenerate'] and os.path.exists(corpus_path):
        logging.info('Load corpus from {}'.format(corpus_path))
        corpus = pickle.load(open(corpus_path, 'rb'))
    else:
        corpus = reader_name(configs)
        logging.info('Save corpus to {}'.format(corpus_path))
        pickle.dump(corpus, open(corpus_path, 'wb'))

    # Define model
    model = model_name(corpus, configs).to(configs['device'])
    logging.info('#params: {}'.format(model.count_variables()))
    logging.info(model)

    # Run model
    data_dict = dict()
    for phase in ['train', 'dev', 'test']:
        data_dict[phase] = model_name.Dataset(model, corpus, phase)
        data_dict[phase].prepare()
    runner = runner_name(configs)
    # logging.info('Test Before Training: ' + runner.print_res(data_dict['test']))
    if configs['load'] > 0:
        model.load_model()
    if configs['train'] > 0:
        runner.train(data_dict)
    eval_res = runner.print_res(data_dict['test'])
    logging.info(os.linesep + 'Test After Training: ' + eval_res)
    # save_rec_results(data_dict['dev'], runner, 100)
    model.actions_after_train()
    logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)


# def save_rec_results(dataset, runner, topk):
#     result_path = os.path.join(args.path, args.dataset, 'rec-{}.csv'.format(init_args.model_name))
#     logging.info('Saving top-{} recommendation results to: {}'.format(topk, result_path))
#     predictions = runner.predict(dataset)  # n_users, n_candidates
#     users, rec_items = list(), list()
#     for i in range(len(dataset)):
#         info = dataset[i]
#         users.append(info['user_id'])
#         item_scores = zip(info['item_id'], predictions[i])
#         sorted_lst = sorted(item_scores, key=lambda x: x[1], reverse=True)[:topk]
#         rec_items.append([x[0] for x in sorted_lst])
#     rec_df = pd.DataFrame(columns=['user_id', 'rec_items'])
#     rec_df['user_id'] = users
#     rec_df['rec_items'] = rec_items
#     rec_df.to_csv(result_path, sep=args.sep, index=False)


def parse_config(config):
    """
    The core configuration file for the entire project.
    Load order: yaml in config -> parser.

    Args:
        config: the collection of all configuration files.

    Returns:

    """
    # First: Load the overall config.
    with open('config/overall.yml', encoding='utf-8') as overall_file:
        overall_data = overall_file.read()
        overall_config = yaml.safe_load(overall_data)
        config.update(overall_config)

    # Second: Load the config of model.
    model_config_path = './config/model/{}.yml'.format(configs['model']['name'])
    if not os.path.exists(model_config_path):
        raise Exception("Please create the yaml file for your model first.")

    with open(model_config_path, encoding='utf-8') as model_config_file:
        model_config_data = model_config_file.read()
        model_config = yaml.safe_load(model_config_data)

    # Third: Load the config of reader and runner
    # Create a new empty dictionary for merging the contents of the two files
    merged_data = {}
    # Add the contents of the first file to the new dictionary
    merged_data.update(config)
    # Add the contents of the second file to the new dictionary, keeping the contents of the first file
    if model_config:
        for key, value in model_config.items():
            if key in merged_data:
                # If the value exists, update it.
                if value:
                    merged_data[key].update(value)
            else:
                merged_data[key] = value
    config.update(merged_data)


if __name__ == '__main__':
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='BPRMF', help='Choose a model to run.')
    init_args, init_extras = init_parser.parse_known_args()

    # init overall configs
    configs = dict()
    configs.update({'model': {}})
    configs.update({'runner': {}})
    configs.update({'reader': {}})

    # model name
    configs['model']['name'] = init_args.model_name

    # load the overall config file
    parse_config(configs)
    print("Present", configs)

    # Args
    # parser = argparse.ArgumentParser(description='')
    # parser = parse_global_args(parser)
    # parser = reader_name.parse_data_args(parser)
    # parser = runner_name.parse_runner_args(parser)
    # parser = model_name.parse_model_args(parser)
    # args, extras = parser.parse_known_args()

    # Dynamic create reader and runner
    model_name = eval('{0}.{0}'.format(init_args.model_name))
    reader_name = eval('{0}'.format(configs['reader']['name']))  # model chooses the reader
    runner_name = eval('{0}'.format(configs['runner']['name']))  # model chooses the runner

    log_args = [configs['model']['name'],
                configs['reader']['dataset'],
                'seed='+str(configs['random_seed']),
                'lr='+str(configs['runner']['lr']),
                'l2='+str(configs['runner']['l2']),
                'batch_size='+str(configs['runner']['batch_size'])]
    log_file_name = '__'.join(log_args).replace(' ', '__')
    if configs['log_file'] == '':
        configs['log_file'] = '../log/{}/{}.txt'.format(configs['model']['name'], log_file_name)
    if configs['model_path'] == '':
        configs['model_path'] = '../model/{}/{}.pt'.format(configs['model']['name'], log_file_name)

    utils.check_dir(configs['log_file'])
    logging.basicConfig(filename=configs['log_file'], level=configs['verbose'])
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(init_args)

    logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
    yaml_str = yaml.dump(configs)
    logging.info(yaml_str)

    main(configs)
