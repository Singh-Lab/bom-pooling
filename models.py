from datasets import *
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(dims[i], dims[i+1]),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(dims[i+1]),
                    nn.Dropout(0.1)
                )
                for i in range(len(dims) - 2)
            ],
            nn.Linear(dims[-2], dims[-1])
        )

    def forward(self, x):
        return self.net(x)

class CrossAttention(nn.Module):
    def __init__(self, **kwargs):
        super(CrossAttention, self).__init__()
        self.qnet = kwargs.get('qnet', MLP([1024, 256, 1024]))
        self.knet = kwargs.get('knet', MLP([1024, 256, 1024]))
        self.vnet = kwargs.get('vnet', MLP([1024, 256, 1024]))
    # a, b are lists
    # a[i] is tensor of shape N^i_k * 1024 where N^i_k = L^i - k + 1 is the no. of k-mers in seq i from set A
    # b[i] is tensor of shape N^i_k * 1024 where N^i_k = L^i - k + 1 is the no. of k-mers in seq i from set B
    # computes avg_dist (
    #   softmax(q(ai) @ k(bi)^T) v(bi) --> cross attention ai, bi
    #   softmax(q(ai) @ k(bi)^T) v(bi) --> cross attention bi, ai
    # )
    def forward(self, a, b):
        cat_a, cat_b = torch.cat(a, dim=0), torch.cat(b, dim=0)
        qa, ka, va = self.qnet(cat_a), self.knet(cat_a), self.vnet(cat_a)
        qb, kb, vb = self.qnet(cat_b), self.knet(cat_b), self.vnet(cat_b)
        idx_a, idx_b = 0, 0
        emb_a, emb_b = [], []
        for i in range(len(a)):
            idx_a_next, idx_b_next = idx_a + a[i].shape[0], idx_b + b[i].shape[0]
            cross_attn_a = F.softmax(
                qa[idx_a: idx_a_next] / 32. @
                kb[idx_b: idx_b_next].transpose(-2, -1),
                dim=-1
            ) @ vb[idx_b: idx_b_next]
            cross_attn_b = F.softmax(
                qb[idx_b: idx_b_next] / 32. @
                ka[idx_a: idx_a_next].transpose(-2, -1),
                dim=-1
            ) @ va[idx_a: idx_a_next]
            emb_a.append(cross_attn_a.mean(dim=0, keepdim=True))
            emb_b.append(cross_attn_b.mean(dim=0, keepdim=True))
            idx_a, idx_b = idx_a_next, idx_b_next
        return emb_a, emb_b


class CrossAttentionKernel(nn.Module):
    def __init__(self, **kwargs):
        super(CrossAttentionKernel, self).__init__()
        self.cross_attention = CrossAttention(**kwargs)

    def forward(self, a, b):
        emb_a, emb_b = self.cross_attention(a, b)
        return torch.cat([- F.cosine_similarity(emb_a[i], emb_b[i]) for i in range(len(a))], dim=0)

# Embedding Layers
class SelfAttentionEmbedding(nn.Module):
    def __init__(self, **kwargs):
        super(SelfAttentionEmbedding, self).__init__()
        self.qnet = kwargs.get('qnet', MLP([1024, 256, 1024]))
        self.knet = kwargs.get('knet', MLP([1024, 256, 1024]))
        self.vnet = kwargs.get('vnet', MLP([1024, 256, 1024]))

    # a is list
    # a[i] is tensor of shape N^i_k * 1024 where N^i_k = L^i - k + 1 is the no. of k-mers in seq i from set A
    def forward(self, a):
        cat_a = torch.cat(a, dim=0)
        qa, ka, va = self.qnet(cat_a), self.knet(cat_a), self.vnet(cat_a)
        idx_a = 0
        res = []
        for i in range(len(a)):
            idx_a_next = idx_a + a[i].shape[0]
            # attn is of shape N^i_k * 1024
            attn = F.softmax(
                qa[idx_a: idx_a_next] / 32. @
                ka[idx_a: idx_a_next].transpose(-1, -2),
                dim=-1
            ) @ va[idx_a: idx_a_next]
            # res[-1] is of shape 1 * 1024
            res.append(attn.mean(dim=0))
        # return batch_size * 1024
        return torch.stack(res, dim=0)


# Multiheaded Linear
class MultiheadLinear(nn.Module):
    def __init__(self, **kwargs):
        super(MultiheadLinear, self).__init__()
        self.qnet = kwargs.get('qnet', MLP([1024, 256, 1024]))
        self.knet = kwargs.get('knet', MLP([1024, 256, 1024]))
        self.vnet = kwargs.get('vnet', MLP([1024, 256, 1024]))

    # a is list
    # a[i] are tensors of shape 1024
    def forward(self, a):
        a = torch.stack(a, dim=0)
        return self.qnet(a) + self.knet(a) + self.vnet(a)
    
class IdentityEmbedding(nn.Module):
    def __init__(self, **kwargs):
        super(IdentityEmbedding, self).__init__()

    def forward(self, a):
        return torch.stack(a, dim=0)


class MultiheadLinearKernel(nn.Module):
    def __init__(self, **kwargs):
        super(MultiheadLinearKernel, self).__init__()
        self.multihead_linear = MultiheadLinear(**kwargs)

    # a, b are lists
    # a[i], b[i] are tensors of shape 1 * 1024
    def forward(self, a, b):
        emb_a, emb_b = self.multihead_linear(a), self.multihead_linear(b)
        return - F.cosine_similarity(emb_a, emb_b)


class MultiheadPreference(nn.Module):
    def __init__(self, **kwargs):
        super(MultiheadPreference, self).__init__()
        self.multihead_linear = MultiheadLinear(**kwargs)

    def forward(self, a, b):
        emb_a, emb_b = self.multihead_linear(a), self.multihead_linear(b)
        return emb_a - emb_b

class IdentityPreference(nn.Module):
    def __init__(self, **kwargs):
        super(IdentityPreference, self).__init__()
    
    def forward(self, a, b):
        a, b = torch.stack(a, dim=0), torch.stack(b, dim=0)
        return a - b

# Simple Euclidean
class EuclideanKernel(nn.Module):
    def __init__(self):
        super(EuclideanKernel, self).__init__()

    # a, b are list
    def forward(self, a, b):
        a, b = torch.cat(a, dim=0), torch.cat(b, dim=0)
        return - F.cosine_similarity(a, b)

class ContrastiveLearner:
    def __init__(self, **kwargs):
        self.data = kwargs['data']
        self.loss = kwargs['contrastive_loss']
        self.history = None
        self.reset_history()

    def reset_history(self):
        raise NotImplementedError

    def evaluate(self, **kwargs):
        raise NotImplementedError

    def train(self, n_epochs=10001, interval=100, batch_size=128, lr=1e-4):
        self.loss.distance_net.to('cuda')
        opt = AdamW(self.loss.distance_net.parameters(), lr=lr)
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-6)
        for ep in range(n_epochs):
            if ep % interval == 0:
                self.evaluate(ep)
            self.loss.train()
            np.random.shuffle(self.data.train_data['anchors'])
            bar = trange(0, self.data.train_data['anchors'].shape[0], batch_size)
            avg_loss = 0.
            for i in bar:
                opt.zero_grad()
                anchor_batch = self.data.train_data['anchors'][i: min(self.data.train_data['anchors'].shape[0], i + batch_size)]    
                anchor, negative, positive = self.data.create_triplet_batch(anchor_batch)                
                loss = self.loss(anchor, positive, negative)
                avg_loss += loss * anchor_batch.shape[0]
                bar.set_description_str(f'Epoch {ep} LR = {scheduler.get_last_lr()} Avg Loss = {avg_loss.item() / self.data.train_data["anchors"].shape[0]:.5f}')
                loss.backward()
                opt.step()
            scheduler.step()

class DPI(ContrastiveLearner):
    def __init__(self, **kwargs):
        super(DPI, self).__init__(**kwargs)
        self.test_pairs = {}
        for fold in TEST_DPI_DOMAINS.keys():
            test_pairs = []
            pos, neg = self.data.test_data[fold]['pos'], self.data.test_data[fold]['neg']
            for i in range(len(pos)):
                for j in pos[i]:
                    test_pairs.append((i, j, 1))
            for i in range(len(neg)):
                for j in neg[i]:
                    test_pairs.append((i, j, 0))
            self.test_pairs[fold] = test_pairs
        
    def reset_history(self):
        self.history = {
            fold: {
                'epochs': [],
                'coeffs': [],
                'acc': [],
                'dist': []
            } for fold in TEST_DPI_DOMAINS
        }
    
    def evaluate(self, epoch, **kwargs):
        batch_size = kwargs.get('eval_batch_size', 256)
        for fold in TEST_DPI_DOMAINS.keys():
            with torch.no_grad():
                self.loss.eval()
                n_test = len(self.test_pairs[fold])
                bar = trange(0, n_test, batch_size)    
                bar.set_description_str(f'Testing {fold}:')
                dist, labels = [], []
                for i in bar:
                    batch_pairs = self.test_pairs[fold][i: min(i + batch_size, n_test)]
                    batch_left, batch_right = [], []
                    for l, r, label in batch_pairs:
                        lemb = self.data.test_data[fold]['dom_emb'][l].to('cuda')
                        remb = self.data.test_data[fold]['pep_emb'][r].to('cuda')
                        batch_left.append(lemb)
                        batch_right.append(remb)
                        labels.append(label)
                    dist.append(self.loss.distance_net(batch_left, batch_right).to('cpu'))
                labels = torch.tensor(labels)
                # dist = (torch.cat(dist, dim=0).flatten() + 1.) / 2.
                dist = torch.sigmoid(2.5 * torch.cat(dist, dim=0).flatten())
                acc = (torch.sum((dist < 0.5) == labels) / dist.shape[0]).item()
                auc = roc_auc_score(labels.numpy(), 1.0 - dist.numpy())
                coeff = torch.corrcoef(torch.stack([dist, labels]))[0, 1].item()
                print(f'Coeff = {coeff:.5f} Acc = {acc:.5f} AUC = {auc:.5f}')
                self.history[fold]['epochs'].append(epoch)
                self.history[fold]['acc'].append(acc)
                self.history[fold]['coeffs'].append(coeff)
                self.history[fold]['dist'].append(dist)
            
class SCOPe(ContrastiveLearner):
    def __init__(self, **kwargs):
        super(SCOPe, self).__init__(**kwargs)
        test_pairs = []
        n_test = self.data.test_data.shape[0]
        for i in range(n_test):
            for j in range(i+1, n_test):
                si, sj = self.data.test_data[i], self.data.test_data[j]
                _, sfi, _, si  = self.data.train_data['meta_info']['membership'][si]
                _, sfj, _, sj  = self.data.train_data['meta_info']['membership'][sj]
                test_pairs.append((si, sj, 1 if sfi == sfj else 0))
        self.test_pairs = test_pairs
        self.n_test = len(self.test_pairs)
    
    def reset_history(self):
        self.history = {
            'epochs': [],
            'coeffs': [],
            'acc': [],
            'dist': []
        }
    
    def evaluate(self, epoch, **kwargs):
        batch_size = kwargs.get('eval_batch_size', 128)
        with torch.no_grad():
            self.loss.eval()
            bar = trange(0, self.n_test, batch_size)    
            bar.set_description_str(f'Testing:')
            dist, labels = [], []
            for i in bar:
                batch_pairs = self.test_pairs[i: min(i + batch_size, self.n_test)]
                batch_left, batch_right = [], []
                for l, r, label in batch_pairs:
                    batch_left.append(self.data.train_data['seq_emb'][l].to('cuda'))
                    batch_right.append(self.data.train_data['seq_emb'][r].to('cuda'))
                    labels.append(label)
                dist.append(self.loss.distance_net(batch_left, batch_right).to('cpu'))
            labels = torch.tensor(labels)
            dist = (torch.cat(dist, dim=0).flatten() + 1.) / 2.
            acc = (torch.sum((dist < 0.5) == labels) / dist.shape[0]).item()
            coeff = torch.corrcoef(torch.stack([dist, labels]))[0, 1].item()
            print(f'Coeff = {coeff:.5f} Acc = {acc:.5f}')
        self.history['epochs'].append(epoch)
        self.history['acc'].append(acc)
        self.history['coeffs'].append(coeff)
        self.history['dist'].append(dist)

class LearnablePooling(nn.Module):
    def __init__(self, pooling_mode, **kwargs):
        super(LearnablePooling, self).__init__()
        self.dim = kwargs['dim']
        self.pooling_mode = pooling_mode
        self.outer_hidden_dim = kwargs.get('outer_hidden_dim', 1024)
        self.qnet = nn.Linear(self.dim, self.outer_hidden_dim)
        self.knet = nn.Linear(self.dim, self.outer_hidden_dim)
        self.vnet = nn.Linear(self.dim, self.outer_hidden_dim)

    def forward(self, x):
        qx, kx, vx = self.qnet(x), self.knet(x), self.vnet(x)
        return pooling[self.pooling_mode](qx + kx + vx)

class BoMPooling(nn.Module):
    def __init__(self, **kwargs):
        super(BoMPooling, self).__init__()
        self.dim = kwargs['dim']
        self.inner_hidden_dim = kwargs.get('inner_hidden_dim', 256)
        self.outer_hidden_dim = kwargs.get('outer_hidden_dim', 1024)
        self.qnet = nn.Linear(self.dim, self.inner_hidden_dim)
        self.knet = nn.Linear(self.dim, self.inner_hidden_dim)
        self.vnet = nn.Linear(self.dim, self.outer_hidden_dim)
        self.k = kwargs['k']
        self.s = kwargs['s']

    def forward(self, x, window=False):
        if window:
            x = x.unfold(1, self.k, self.s).mean(dim=-1)
        qx = self.qnet(x)
        kx, vx = self.knet(x), self.vnet(x)
        attn = F.softmax(qx @ kx.transpose(-1, -2) / (self.inner_hidden_dim ** .5), dim=-1) @ vx
        return attn.mean(dim=1)

class LLMPredictor(nn.Module):
    def __init__(self, **kwargs):
        super(LLMPredictor, self).__init__()
        self.pooling_mode = kwargs['pooling_mode']
        self.trainable = kwargs.get('trainable', False)
        if self.pooling_mode == 'bom':
            self.pooling = BoMPooling(
                dim=PLM_dim[kwargs['emb_model']], 
                k=kwargs['k'], s=kwargs['s'],
                inner_hidden_dim=256,
                outer_hidden_dim=1024
            ).to('cuda')
            self.fc = MLP([1024, 512, 256, 1]).to('cuda')
        elif self.trainable:
            self.qnet = nn.Linear(PLM_dim[kwargs['emb_model']], 256).to('cuda')
            self.knet = nn.Linear(PLM_dim[kwargs['emb_model']], 256).to('cuda')
            self.vnet = nn.Linear(PLM_dim[kwargs['emb_model']], 1024).to('cuda')
            self.fc1 = MLP([1024, 512, 256]).to('cuda')
            self.fc = MLP([256, 1]).to('cuda')
            print(f'Trainable Pooling')
        else:
            self.fc = MLP([PLM_dim[kwargs['emb_model']], 512, 256, 1]).to('cuda')
     
    # list of seq
    def forward(self, x):
        if self.pooling_mode == 'bom':
            x = self.pooling(x)
        elif self.trainable:
            x = F.relu(self.fc1(self.vnet(x)) + self.qnet(x) + self.knet(x))
        return self.fc(x)