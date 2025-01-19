import os
import random

import torch
import numpy as np

from time import time
from tqdm import tqdm
from copy import deepcopy
import logging
from prettytable import PrettyTable

from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate import test
from utils.helper import early_stopping

from torch.utils.tensorboard import writer

n_users = 0
n_items = 0


def get_feed_dict(train_entity_pairs, train_pos_set, start, end, n_negs=1):

    def sampling(user_item, train_set, n):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            negitems = []
            for i in range(n):  # sample n times
                while True:
                    negitem = random.choice(range(n_items))
                    if negitem not in train_set[user]:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                       train_pos_set,
                                                       n_negs*K)).to(device)
    return feed_dict


def temperature_scaled_softmax(logits, temperature, dim):
    logits = logits / temperature
    return torch.softmax(logits, dim=dim)

if __name__ == '__main__':
    """fix the random seed"""
    seed = 2019
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, user_dict, n_params, norm_mat = load_data(args)
    train_cf_size = len(train_cf)
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_negs = args.n_negs
    K = args.K

    """define model"""
    # MF can be implemented by setting the number of layers of LightGCN to 0.
    from modules.LightGCN import LightGCN
    if args.gnn == 'lightgcn':
        model = LightGCN(n_params, args, norm_mat).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    # Initialize the loss matrix to store losses
    # put the loss matrix in cpu
    loss_mat = torch.full((n_users, n_items), float('inf')).to('cpu')

    print("start training ...")
    for epoch in range(args.epoch):
        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)

        """training"""
        model.train()
        loss, s = 0, 0
        hits = 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_,
                                  user_dict['train_user_set'],
                                  s, s + args.batch_size,
                                  n_negs)

            user = batch['users']
            item = batch['pos_items']

            if epoch <= 1:
                weights = None
            else:
                stacked_factor = torch.stack([ui_factor[user.to('cpu'), item.to('cpu')], u_factor[user.to('cpu')], i_factor[item.to('cpu')]], dim=1)
                # stacked_factor = torch.stack([ui_factor[user.to('cpu'), item.to('cpu')], i_factor[item.to('cpu')]], dim=1)
                # stacked_factor = torch.stack([ui_factor[user.to('cpu'), item.to('cpu')], u_factor[user.to('cpu')]], dim=1)
                # stacked_factor = torch.stack([u_factor[user.to('cpu')], i_factor[item.to('cpu')]], dim=1)
                # stacked_factor = torch.stack([ui_factor[user.to('cpu'), item.to('cpu')]], dim=1)
                # stacked_factor = torch.stack([u_factor[user.to('cpu')]], dim=1)
                # stacked_factor = torch.stack([i_factor[item.to('cpu')]], dim=1)
                

                # weights = torch.max(stacked_factor, dim=1)[0].to(device)
                # weights = torch.min(stacked_factor, dim=1)[0].to(device)
                weights = torch.clamp(torch.mean(stacked_factor, dim=1).to(device), max=1.0)

                # For debugging
                # print('############ui_factor##############')
                # print(ui_factor[user.to('cpu'), item.to('cpu')])
                # print('############u_factor##############')
                # print(u_factor[user.to('cpu')])
                # print('############i_factor##############')
                # print(i_factor[item.to('cpu')])

            batch_loss, pos_loss_wo_mean, _ = model(batch, weights=weights)

            with torch.no_grad():
                # Update the loss matrix with batch_loss
                if epoch <= 1:
                    loss_mat[user.to('cpu'), item.to('cpu')] = pos_loss_wo_mean.to('cpu')
                else:
                    # Smoothing loss to overcome dimension oscillation problem
                    loss_mat[user.to('cpu'), item.to('cpu')] = args.a * loss_mat[user.to('cpu'), item.to('cpu')] + (1 - args.a) * pos_loss_wo_mean.to('cpu')

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.batch_size

        user_int_count = torch.sum(loss_mat != float('inf'), dim=1)
        item_int_count = torch.sum(loss_mat != float('inf'), dim=0)

        ui_factor =  temperature_scaled_softmax(-loss_mat, args.t, dim=1) * user_int_count.unsqueeze(dim=1) # multiply counts to make ui_reflec_factor even

        user_loss = torch.sum(((loss_mat != float('inf')) * loss_mat).nan_to_num(), dim=1) / user_int_count
        # u_factor = temperature_scaled_softmax(-user_loss, args.t, dim=0) * n_users
        u_factor = temperature_scaled_softmax(-torch.nan_to_num(user_loss, nan = float('inf')), args.t, dim=0) * torch.count_nonzero(user_int_count)

        item_loss = torch.sum(((loss_mat != float('inf')) * loss_mat).nan_to_num(), dim=0) / item_int_count
        i_factor = temperature_scaled_softmax(-torch.nan_to_num(item_loss, nan = float('inf')), args.t, dim=0) * torch.count_nonzero(item_int_count) # some items may no interacted
        
        train_e_t = time()

        if epoch % 5 == 0:
            """testing"""

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time(s)", "tesing time(s)", "Loss", "recall", "ndcg"]

            model.eval()
            test_s_t = time()
            test_ret = test(model, user_dict, n_params, mode='test')
            test_e_t = time()
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), test_ret['recall'], test_ret['ndcg']])

            if user_dict['valid_user_set'] is None:
                valid_ret = test_ret
            else:
                test_s_t = time()
                valid_ret = test(model, user_dict, n_params, mode='valid')
                test_e_t = time()
                train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), valid_ret['recall'], valid_ret['ndcg']])
            print(train_res)

            file_path = './logs/' + str(args.context_hops) + args.dataset + '_' + str(args.t) + '_' + str(args.a) + '.txt'
            if not os.path.exists(file_path):
                with open(file_path, 'w'):  # Create an empty file
                    pass
            with open(file_path, 'a') as file:
                file.write(str(train_res) + '\n')

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for 10 successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(valid_ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=10)
            if should_stop:
                break

            """save weight"""
            if valid_ret['recall'][0] == cur_best_pre_0 and args.save:
                torch.save(model.state_dict(), args.out_dir + 'model_' + '.ckpt')
        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4fs, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()))

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))
