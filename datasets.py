from utils import *
from plm_wrapper import *
from tape.datasets import LMDBDataset
from torchdrug import datasets
from torchdrug import transforms
from DeepProtein.dataset import *

DATA_ROOT = "/scratch/gpfs/qh1002/data"
DATA_DIR = {
    'fluorescence': f'{DATA_ROOT}/fluorescence',
    'beta_lactamase': f'{DATA_ROOT}/beta_lactamase',
    'dpi': f'{DATA_ROOT}/protein-peptide/data',
    'scope': f'{DATA_ROOT}/SCOPe'
}
DPI_DOMAINS = [
    'PTB', 
    'PTP', 
    'Kinase_TK', 
]
TEST_DPI_DOMAINS = {
    'PTB': 13619,
    'PTP': 69252,
    'Kinase_TK': 136229
}
DEEP_PROTEIN_ROOT = '/home/qh1002/Code/DeepProtein/data'

class RegressionDataset:
    def __init__(self, **kwargs):
        self.dataset = kwargs.get('dataset', 'fluorescence')
        self.emb_model = kwargs.get('emb_model', 'protbert')
        self.batch_size = kwargs.get('embedding_batch_size', 128)
        self.pooling_mode = kwargs.get('pooling_mode', 'avg')
        self.save = kwargs.get('save', 'load_pool')
        raw_data_dir, pool_data_dir = self.data_dir(**kwargs)
        print(f'Loading {self.dataset} dataset')
        if self.save == 'save':
            self.emb_model = PLM[kwargs.get('emb_model', 'prottrans')]()
            self.embed_data(**kwargs)
            torch.save(self.make_data_dict(), raw_data_dir)
            self.filter_data(**kwargs)
            self.pool_data(**kwargs)
            torch.save(self.make_data_dict(), pool_data_dir)
        elif self.save == 'load_raw_only':
            self.load_data_dict(torch.load(raw_data_dir))
            self.filter_data(**kwargs)
        elif self.save == 'load_raw':
            self.load_data_dict(torch.load(raw_data_dir))
            self.filter_data(**kwargs)
            self.pool_data(**kwargs)
            torch.save(self.make_data_dict(), pool_data_dir)
        elif self.save == 'load_pool':
            self.load_data_dict(torch.load(pool_data_dir))
    
    def data_dir(self, **kwargs):
        suffix = f'_k{kwargs["k"]}_s{kwargs["stride"]}' if self.pooling_mode == 'bom' else ''
        emb_model = kwargs.get('emb_model', 'prottrans')
        raw_data_dir = f'{DATA_DIR[self.dataset]}/{self.dataset}_{emb_model}.pt'
        pool_data_dir = f'{DATA_DIR[self.dataset]}/{self.dataset}_{emb_model}_{self.pooling_mode}{suffix}.pt'
        return raw_data_dir, pool_data_dir
    
    def make_data_dict(self):
        return {
            'x_train': self.x_train,
            'y_train': self.y_train,
            'x_test': self.x_test,
            'y_test': self.y_test
        }

    def embed_data(self, **kwargs):
        train_data = FluorescenceDataset(deep_prot_data_dir, 'train') if self.dataset == 'fluorescence' else Beta_lactamase(deep_prot_data_dir, 'train')
        test_data = FluorescenceDataset(deep_prot_data_dir, 'test') if self.dataset == 'fluorescence' else Beta_lactamase(deep_prot_data_dir, 'test')
        self.x_train, self.y_train, _ = collate_fn(train_data)
        self.x_test, self.y_test, _ = collate_fn(test_data)
        def embed(fold_x):
            n_data = len(fold_x)
            emb = []
            bar = trange(0, n_data, bs)
            with torch.no_grad():
                for i in bar:
                    x_batch = fold_x[i: min(n_data, i + bs)]
                    x_batch = emb_model(x_batch)
                    if pooling_mode == 'bom':
                        if (k==1) and (s==1):
                            emb.append(x_batch.cpu())
                        else:    
                            emb.append(x_batch.unfold(1, k, s).mean(-1).cpu())
                    else:
                        emb.append(pooling[pooling_mode](x_batch).cpu())
            return torch.cat(emb, dim=0)
        self.x_train = embed(self.x_train)
        self.x_test = embed(self.x_test)
        
    def pool_data(self, **kwargs):
        pooling_mode = kwargs['pooling_mode']
        if pooling_mode == 'cls':
            self.cls_pooling()
        if pooling_mode == 'eos':
            self.eos_pooling()
        if pooling_mode == 'avg':
            self.avg_pooling()
        if pooling_mode == 'bom':
            self.bom_pooling(kwargs['k'], kwargs['stride'])

    def cls_pooling(self):
        print(f'Performing CLS Pooling')
        self.x_train = [x[0, :] for x in tqdm(self.x_train)]
        self.x_test = [x[0, :] for x in tqdm(self.x_test)]
        gc.collect()
    
    def cls_pooling(self):
        print(f'Performing EOS Pooling')
        self.x_train = [x[-1, :] for x in tqdm(self.x_train)]
        self.x_test = [x[-1, :] for x in tqdm(self.x_test)]
        gc.collect()

    def avg_pooling(self):
        print(f'Performing Avg Pooling')
        self.x_train = [x[1:-1, :].mean(dim=0) for x in tqdm(self.x_train)]
        self.x_test = [x[1:-1, :].mean(dim=0) for x in tqdm(self.x_test)]
        gc.collect()

    def bom_pooling(self, k, stride):
        print(f'Performing BoM Pooling')
        self.x_train = [x[1:-1, :].unfold(0, k, stride).mean(dim=-1) for x in tqdm(self.x_train)]
        self.x_test = [x[1:-1, :].unfold(0, k, stride).mean(dim=-1) for x in tqdm(self.x_test)]
        gc.collect()


class PairedDataset:
    def __init__(self, **kwargs):
        self.emb_model = kwargs.get('emb_model', 'protbert')
        self.pooling_mode = kwargs.get('pooling_mode', 'avg')
        self.save = kwargs.get('save', 'load_pool')
        data_dir = self.data_dir(**kwargs)
        
        if self.save == 'save':
            self.emb_net = PLM[self.emb_model]()
            self.make_train(**kwargs)
            self.make_test(**kwargs)
            torch.save(self.train_data, data_dir['raw_train'])
            torch.save(self.test_data, data_dir['raw_test'])
            self.pool_data(**kwargs)
            torch.save(self.train_data, data_dir['pool_train'])
            torch.save(self.test_data, data_dir['pool_test'])
        elif self.save == 'load_raw': 
            self.train_data = torch.load(data_dir['raw_train'], map_location='cpu')
            self.test_data = torch.load(data_dir['raw_test'], map_location='cpu')
            self.pool_data(**kwargs)
            torch.save(self.train_data, data_dir['pool_train'])
            torch.save(self.test_data, data_dir['pool_test'])
        elif self.save == 'load_raw_only':
            self.train_data = torch.load(data_dir['raw_train'], map_location='cpu')
            self.test_data = torch.load(data_dir['raw_test'], map_location='cpu')
        elif self.save == 'load_pool':
            self.train_data = torch.load(data_dir['pool_train'], map_location='cpu')
            self.test_data = torch.load(data_dir['pool_test'], map_location='cpu')
    
    def data_dir(self, **kwargs):
        raise NotImplementedError
    
    def make_train(self, **kwargs):
        raise NotImplementedError
    
    def make_test(self, **kwargs):
        raise NotImplementedError

    def pool_data(self, **kwargs):
        raise NotImplementedError
    
    def create_triplet_batch(self, anchor):
        raise NotImplementedError

class DPIDataset(PairedDataset):
    def __init__(self, **kwargs):
        super(DPIDataset, self).__init__(**kwargs)
    
    def data_dir(self, **kwargs):
        data_dir = {}
        suffix = f'_k{kwargs["k"]}_s{kwargs["stride"]}' if self.pooling_mode == 'bom' else ''
        data_dir['raw_train'] = f'{DATA_DIR["dpi"]}/train_{self.emb_model}.pt'
        data_dir['pool_train'] = f'{DATA_DIR["dpi"]}/train_{self.emb_model}_{self.pooling_mode}{suffix}.pt'
        data_dir['raw_test'] = f'{DATA_DIR["dpi"]}/test_{self.emb_model}.pt'
        data_dir['pool_test'] = f'{DATA_DIR["dpi"]}/test_{self.emb_model}_{self.pooling_mode}{suffix}.pt'
        return data_dir

    def make_train(self, **kwargs):
        print(f'Preprocessing training data')
        train_data = {}
        train_data['doms'], train_data['peps'], train_data['pos'], train_data['neg'] = self.load_paired_data('train', minlen=kwargs['minlen'])
        with torch.no_grad():
            train_data['dom_emb'] = [self.emb_net([d]) for d in tqdm(train_data['doms'])]
            train_data['pep_emb'] = [self.emb_net([p]) for p in tqdm(train_data['peps'])]
        train_data['anchors'] = np.array([
            i for i in range(len(train_data['doms']))
            if len(train_data['pos'][i]) and len(train_data['neg'][i])
        ])
        self.train_data = train_data
        

    def make_test(self, **kwargs):
        test_data = {}
        for test_fold in DPI_DOMAINS:
            print(f'Preprocessing test data in {test_fold} domain')
            fold_data = {}
            fold_data['doms'], fold_data['peps'], fold_data['pos'], fold_data['neg'] = self.load_paired_data(test_fold, minlen=kwargs['minlen'])
            with torch.no_grad():
                fold_data['dom_emb'] = [self.emb_net([d]) for d in tqdm(fold_data['doms'])]
                fold_data['pep_emb'] = [self.emb_net([p]) for p in tqdm(fold_data['peps'])]
            test_data[test_fold] = fold_data
        self.test_data = test_data

    def load_paired_data(self, fold, minlen=0):
        if fold == 'train':
            f = open(f'{DATA_DIR["dpi"]}/data_without_processed_duplicates/preprocessed_raw_data.csv')
        else:
            f = open(f'{DATA_DIR["dpi"]}/data_without_processed_duplicates/raw_data/{fold}.csv')
        lines = f.readlines()[0 if fold == 'train' else 1:]
        doms, peps = set(), set()
        for line in lines:
            tokens = line.split(',')
            dom, pep = tokens[1], tokens[3 if fold == 'train' else 2]
            dom, pep = re.sub(r"[-y]", "", dom), re.sub(r"[-y]", "", pep)
            if len(dom) > minlen:
                doms.add(dom)
                peps.add(pep)
                
        doms, peps = list(doms), list(peps)
        dom_id = {d: i for i, d in enumerate(doms)}
        pep_id = {p: i for i, p in enumerate(peps)}
        pos = [set() for _ in range(len(doms))]
        neg = [set() for _ in range(len(doms))]
        for line in lines:
            tokens = line.split(',')
            dom, pep, label = tokens[1], tokens[3 if fold == 'train' else 2], int(tokens[-1])
            dom, pep = re.sub(r"[-y]", "", dom), re.sub(r"[-y]", "", pep)
            if len(dom) > minlen:
                (pos if label else neg)[dom_id[dom]].add(pep_id[pep])
        return doms, peps, [list(p) for p in pos], [list(n) for n in neg]

    def pool_data(self, **kwargs):
        print(f'Performing Avg Pooling for short peptide chains')
        self.train_data['pep_emb'] = [x[0, 1:-1, :].mean(dim=0, keepdim=(self.pooling_mode=='bom')) for x in tqdm(self.train_data['pep_emb'])]
        for test_fold in DPI_DOMAINS:
            self.test_data[test_fold]['pep_emb'] = [x[0, 1:-1, :].mean(dim=0, keepdim=(self.pooling_mode=='bom')) for x in tqdm(self.test_data[test_fold]['pep_emb'])]

        if kwargs['pooling_mode'] == 'cls':
            print(f'Performing CLS Pooling for domains')
            self.train_data['dom_emb'] = [x[0, 0, :] for x in tqdm(self.train_data['dom_emb'])]
            for test_fold in DPI_DOMAINS:
                self.test_data[test_fold]['dom_emb'] = [x[0, 0, :] for x in tqdm(self.test_data[test_fold]['dom_emb'])]
                gc.collect()

        if kwargs['pooling_mode'] == 'sep':
            print(f'Performing SEP Pooling for domains')
            self.train_data['dom_emb'] = [x[0, -1, :] for x in tqdm(self.train_data['dom_emb'])]
            for test_fold in DPI_DOMAINS:
                self.test_data[test_fold]['dom_emb'] = [x[0, -1, :] for x in tqdm(self.test_data[test_fold]['dom_emb'])]
                gc.collect()

        if kwargs['pooling_mode'] == 'avg':
            print(f'Performing Avg Pooling for domains')
            self.train_data['dom_emb'] = [x[0, 1:-1, :].mean(dim=0) for x in tqdm(self.train_data['dom_emb'])]
            for test_fold in DPI_DOMAINS:
                self.test_data[test_fold]['dom_emb'] = [x[0, 1:-1, :].mean(dim=0) for x in tqdm(self.test_data[test_fold]['dom_emb'])]
                gc.collect()

        if kwargs['pooling_mode'] == 'bom':
            print(f'Performing BoM Pooling for domains')
            self.train_data['dom_emb'] = [
                x[0, 1:-1, :].unfold(0, min(kwargs['k'], x.shape[1]-2), kwargs['stride']).mean(dim=-1) 
                for x in tqdm(self.train_data['dom_emb'])
            ]
            for test_fold in DPI_DOMAINS:
                self.test_data[test_fold]['dom_emb'] = [
                    x[0, 1:-1, :].unfold(0, min(kwargs['k'], x.shape[1]-2), kwargs['stride']).mean(dim=-1) 
                    for x in tqdm(self.test_data[test_fold]['dom_emb'])
                ]
                gc.collect()
        
    def create_triplet_batch(self, anchor):
        pos_emb = [self.train_data['pep_emb'][np.random.choice(self.train_data['pos'][a])].to('cuda') for a in anchor]
        neg_emb = [self.train_data['pep_emb'][np.random.choice(self.train_data['neg'][a])].to('cuda') for a in anchor]
        anchor_emb = [self.train_data['dom_emb'][a].to('cuda') for a in anchor]
        return anchor_emb, neg_emb, pos_emb

class SCOPeDataset(PairedDataset):
    def __init__(self, **kwargs):
        super(SCOPeDataset, self).__init__(**kwargs)
        self.train_adj_list = torch.load(f'{DATA_DIR["scope_v2"]}/class_{kwargs["cls"]}/train_scope_{kwargs["cls"]}_adjlist.pt')
        self.train_data['anchors'] = np.array([
                i for i in range(len(self.train_adj_list)) 
                if (len(self.train_adj_list[i]) > 1)
                and (len(self.train_adj_list[i]) < 200)
            ]
        )
        self.test_adj_list = torch.load(f'{DATA_DIR["scope_v2"]}/class_{kwargs["cls"]}/test_scope_{kwargs["cls"]}_adjlist.pt')

    def data_dir(self, **kwargs):
        data_dir = {}
        suffix = f'_k{kwargs["k"]}_s{kwargs["stride"]}' if self.pooling_mode == 'bom' else ''
        data_dir['raw_train'] = f'{DATA_DIR["scope_v2"]}/class_{kwargs["cls"]}/train_{self.emb_model}.pt'
        data_dir['pool_train'] = f'{DATA_DIR["scope_v2"]}/class_{kwargs["cls"]}/train_{self.emb_model}_{self.pooling_mode}{suffix}.pt'
        data_dir['raw_test'] = f'{DATA_DIR["scope_v2"]}/class_{kwargs["cls"]}/test_{self.emb_model}.pt'
        data_dir['pool_test'] = f'{DATA_DIR["scope_v2"]}/class_{kwargs["cls"]}/test_{self.emb_model}_{self.pooling_mode}{suffix}.pt'
        return data_dir
    
    def make_fold(self, fold, emb_batch_size, **kwargs):
        raw = torch.load(f'{DATA_DIR["scope_v2"]}/class_{kwargs["cls"]}/{fold}_scope_data_{kwargs["cls"]}.pt')
        emb = []
        n = len(raw['seq_list'])
        with torch.no_grad():
            for i in trange(0, n, emb_batch_size):
                batch_id = raw['seq_list'][i: min(i + emb_batch_size, n)]
                batch = [raw['seq_dict'][sf][fa][j] for sf, fa, j in batch_id]
                batch_emb = self.emb_net(batch).to('cpu')
                emb += [batch_emb[j, :len(batch[j]) + 2, :] for j in range(batch_emb.shape[0])]
                gc.collect()  
        return emb, raw['seq_list']

    def make_train(self, **kwargs):
        emb, seq_list = self.make_fold('train', 64, **kwargs)
        self.train_data = {'emb': emb, 'seq_list': seq_list}
    
    def make_test(self, **kwargs):
        emb, seq_list = self.make_fold('test', 64, **kwargs)
        self.test_data = {'emb': emb, 'seq_list': seq_list}

    
    def pool_data(self, **kwargs):
        if kwargs['pooling_mode'] == 'cls':
            print(f'Performing CLS Pooling')
            self.train_data['emb'] = [x[0, :] for x in tqdm(self.train_data['emb'])]
            self.test_data['emb'] = [x[0, :] for x in tqdm(self.test_data['emb'])]

        if kwargs['pooling_mode'] == 'avg':
            print(f'Performing Avg Pooling')
            self.train_data['emb'] = [x[1:-1, :].mean(dim=0) for x in tqdm(self.train_data['emb'])]
            self.test_data['emb'] = [x[1:-1, :].mean(dim=0) for x in tqdm(self.test_data['emb'])]

        if kwargs['pooling_mode'] == 'sep':
            print(f'Performing SEP Pooling')
            self.train_data['emb'] = [x[-1, :] for x in tqdm(self.train_data['emb'])]
            self.test_data['emb'] = [x[-1, :] for x in tqdm(self.test_data['emb'])]

        if kwargs['pooling_mode'] == 'bom':
            print(f'Performing BoM Pooling')
            self.train_data['emb'] = [x[1:-1, :].unfold(0, kwargs['k'], kwargs['stride']).mean(dim=-1) for x in tqdm(self.train_data['emb'])]
            self.test_data['emb'] = [x[1:-1, :].unfold(0, kwargs['k'], kwargs['stride']).mean(dim=-1) for x in tqdm(self.test_data['emb'])]

        gc.collect()

    def create_triplet_batch(self, anchor):
        anchor_emb, neg_emb, pos_emb = [], [], []
        for idx in anchor:
            anchor_emb.append(self.train_data['emb'][idx].to('cuda'))
            
            # Positive sampling
            pos_idx = np.random.choice(self.train_adj_list[idx])
            pos_emb.append(self.train_data['emb'][pos_idx].to('cuda'))
            
            # Negative sampling
            neg_idx = pos_idx
            while neg_idx in self.train_adj_list[idx]:
                neg_idx = np.random.choice(len(self.train_adj_list))
            neg_emb.append(self.train_data['emb'][neg_idx].to('cuda'))
                
        return anchor_emb, neg_emb, pos_emb

def generate_training_data(task, k, stride):
    pooling_mode = [
        'bom', 
        'sep',
        'cls', 
        'avg'
    ]
    embedding_model = [
        # 'protbert', 
        # 'prottrans', 
        # 'esm2-35M', 
        # 'esm2-150M',
        'esm2-650M'
    ]
    for em in embedding_model:
        for pm in pooling_mode:
            data = None
            if task in ['fluorescence', 'beta_lactamase']:
                data = TapeRegressionDataset(
                    dataset=task,
                    emb_model=em,
                    save='save' if pm == 'bom' else 'load_raw',
                    pooling_mode=pm,
                    k=k, stride=stride
                )
            elif task == 'scope':
                data = SCOPeDataset(
                    emb_model=em,
                    save='save' if pm == 'bom' else 'load_raw',
                    pooling_mode=pm,
                    k=k, stride=stride,
                    cls='all'
                )
            elif task == 'dpi':
                data = DPIDataset(
                    emb_model=em,
                    save='save' if pm == 'bom' else 'load_raw',
                    pooling_mode=pm,
                    k=k, stride=stride,
                    minlen=0,
                )

if __name__ == '__main__':
    torch.cuda.set_device(0)
    generate_training_data('fluorescence', 100, 20)
    # generate_training_data('beta_lactamase', 100, 20)
    # generate_training_data('dpi', 40, 8)
    # generate_training_data('scope_v2', 100, 80)