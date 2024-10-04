import argparse
import logging
import os.path

import numpy as np
import torch
import torch.utils.data as Data

from data_preprocess.prepare_dataset import preprocess
from model.DKT import DKT


def get_data_loader(data_path, batch_size, shuffle=False):
    data = torch.FloatTensor(np.load(data_path))
    data_loader = Data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def run(args):
    device = torch.device(('cuda:%d' % args.gpu) if torch.cuda.is_available() else 'cpu')

    # 准备数据
    if (not os.path.exists(f'data/{args.dataset}/train_data.npy') or
            not os.path.exists(f'data/{args.dataset}/test_data.npy')):
        preprocess(args)
    train_loader = get_data_loader(f'data/{args.dataset}/train_data.npy', args.batch_size, True)
    test_loader = get_data_loader(f'data/{args.dataset}/test_data.npy', args.batch_size, False)

    logging.getLogger().setLevel(logging.INFO)
    logging.info('Data loaded successfully.')

    logging.info('Start training...')
    # 创建模型
    dkt = DKT(args.num_concepts, args.hidden_size, args.num_layers, device=device)
    # 对模型进行训练和保存
    dkt.train(train_data=train_loader, test_data=test_loader,
              save_model_file=os.path.join(args.save_root, args.dataset, args.save_model_file), epoch=args.epoch,
              lr=args.lr)
    logging.info('Training finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use.')
    parser.add_argument('--max_len', type=int, default=200, help='Maximum length of the sequence.')
    parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle the data.')
    parser.add_argument('--train_size', type=float, default=0.8, help='Ratio of training data.')

    # 数据集相关设置
    parser.add_argument('--dataset', type=str, default='c_filter2', help='Dataset in the folder named "data".')
    parser.add_argument('--num_concepts', type=int, default=116, help='Number of concepts.')

    # 网络相关设置
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers.')
    parser.add_argument('--epoch', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate.')

    # 保存相关设置
    parser.add_argument('--save_model_file', type=str, default='model', help='File to save the model.')
    parser.add_argument('--save_root', type=str, default='saved_models', help='Whether to shuffle the data.')

    run(parser.parse_args())
