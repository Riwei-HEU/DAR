import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, n_hops, n_users, interact_mat,
                 edge_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.interact_mat = interact_mat
        self.n_users = n_users
        self.n_hops = n_hops
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_embed, item_embed,
                mess_dropout=True, edge_dropout=True):
        # user_embed: [n_users, channel]
        # item_embed: [n_items, channel]

        # all_embed: [n_users+n_items, channel]
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        agg_embed = all_embed
        embs = [all_embed]

        for hop in range(self.n_hops):
            interact_mat = self._sparse_dropout(self.interact_mat,
                                                self.edge_dropout_rate) if edge_dropout \
                else self.interact_mat

            agg_embed = torch.sparse.mm(interact_mat, agg_embed)
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            # agg_embed = F.normalize(agg_embed)
            embs.append(agg_embed)
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
        return embs[:self.n_users, :], embs[self.n_users:, :]


class LightGCN(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(LightGCN, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.edge_dropout = args_config.edge_dropout
        self.edge_dropout_rate = args_config.edge_dropout_rate
        self.pool = args_config.pool
        self.simi = args_config.simi
        self.epoch = args_config.epoch
        self.K = args_config.K

        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")

        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)
        
        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))

        # [n_users+n_items, n_users+n_items]
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         interact_mat=self.sparse_norm_adj,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, batch=None, weights=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']  # [batch_size, n_negs * K]

        # user_gcn_emb: [n_users, channel]
        # item_gcn_emb: [n_users, channel]
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)

        neg_gcn_embs = item_gcn_emb[neg_item[:, :self.K]]

        return self.create_bce_loss(user, user_gcn_emb[user], item_gcn_emb[pos_item], neg_gcn_embs, weights)

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def similarity(self, user_embeddings, item_embeddings):
        # [-1, n_hops, channel]
        if self.simi == 'ip':
            return (user_embeddings * item_embeddings).sum(dim=-1)
        elif self.simi == 'cos':
            return F.cosine_similarity(user_embeddings, item_embeddings, dim=-1)
        elif self.simi == 'ed':
            return ((user_embeddings - item_embeddings) ** 2).sum(dim=-1)
        else:  # ip
            return (user_embeddings * item_embeddings).sum(dim=-1)

    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed, self.item_embed, edge_dropout=False, mess_dropout=False)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())
    
    def _adjust_dim(self, embeddings, users, dims_to_use):
        adjusted_embeddings = []
        for i, user_idx in enumerate(users):
            dim = dims_to_use[i].item()
            adjusted_embeddings.append(embeddings[user_idx][:dim].unsqueeze(0))
        return torch.cat(adjusted_embeddings, dim=0)

    def create_bpr_loss(self, user, user_gcn_emb, pos_gcn_embs, neg_gcn_embs, weights):
        # user_gcn_emb: [batch_size, n_hops+1, channel]
        # pos_gcn_embs: [batch_size, n_hops+1, channel]
        # neg_gcn_embs: [batch_size, K, n_hops+1, channel]
        
        batch_size = user.shape[0]
        
        u_e = self.pooling(user_gcn_emb)
        pos_e = self.pooling(pos_gcn_embs)
        neg_e = self.pooling(neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(batch_size, self.K, -1)
        
        if weights is not None:
            dims = (weights * self.emb_size).to(torch.long)
            index = torch.arange(self.emb_size, device=self.device).unsqueeze(0).expand(batch_size, -1)
            mask = index < dims.unsqueeze(1)

            u_e = u_e * mask
            pos_e = pos_e * mask

        pos_scores = (u_e * pos_e).sum(dim=-1)
        neg_scores = (u_e.unsqueeze(dim=1) * neg_e).sum(dim=-1)
        
        mf_loss = torch.log(1 + torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1))
        
        # cul regularizeraf
        regularize = (torch.norm(user_gcn_emb[:, 0, :]) ** 2
                      + torch.norm(pos_gcn_embs[:, 0, :]) ** 2
                      + torch.norm(neg_gcn_embs[:, :, 0, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size

        return torch.mean(mf_loss) + emb_loss, mf_loss, emb_loss

    def create_bce_loss(self, user, user_gcn_emb, pos_gcn_embs, neg_gcn_embs, weights=None):
        # user_gcn_emb: [batch_size, n_hops+1, channel]
        # pos_gcn_embs: [batch_size, n_hops+1, channel]
        # neg_gcn_embs: [batch_size, K, n_hops+1, channel]
        
        batch_size = user.shape[0]
        
        u_e = self.pooling(user_gcn_emb)
        pos_e = self.pooling(pos_gcn_embs)
        neg_e = self.pooling(neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(batch_size, self.K, -1)

        if weights is not None:
            dims = (weights * self.emb_size).to(torch.long)
            index = torch.arange(self.emb_size, device=self.device).unsqueeze(0).expand(batch_size, -1)
            mask = index < dims.unsqueeze(1)

            u_e = u_e * mask
            pos_e = pos_e * mask
        
        pos_scores = (u_e * pos_e).sum(dim=-1)
        neg_scores = (u_e.unsqueeze(dim=1) * neg_e).sum(dim=-1)

        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)

        # Compute BCE loss for positive and negative examples separately
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, reduction='none')
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, reduction='none').sum(dim=1)

        # Combine the losses
        mf_loss = pos_loss + neg_loss
        
        # cul regularizer
        regularize = (torch.norm(user_gcn_emb[:, 0, :]) ** 2
                      + torch.norm(pos_gcn_embs[:, 0, :]) ** 2
                      + torch.norm(neg_gcn_embs[:, :, 0, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size

        return mf_loss.mean() + emb_loss, pos_loss, emb_loss
