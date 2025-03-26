from utils import *
from datasets import *

scope_dir = f'{DATA_ROOT}/SCOPe'

def download_scope(structural_class='a'):
    all_domains = open(f'{scope_dir}/dir.cla.scope.2.08-stable.txt').readlines()[4:]
    pids = set()
    for dom in all_domains:
        tokens = dom.strip().split('\t')
        if tokens[3].split('.')[0] == structural_class:
            pids.add(tokens[1])

    seqs = {}
    for i in range(100):
        if len(pids) == 0:
            break
        print(f'Attempting to download {len(pids)} seqs from PDB')
        success, failure = download_pdb(pids)
        seqs.update(success)
        
        torch.save(seqs, f'{scope_dir}/class_{structural_class}/{structural_class}_seqs.pt')
        if len(failure) == len(pids):
            print(len(failure))
            break
        pids = failure

def annotate_scope(structural_class='a', retry=False):
    all_domains = open(f'{scope_dir}/dir.cla.scope.2.08-stable.txt').readlines()[4:]
    all_available_seqs = torch.load(f'{scope_dir}/class_{structural_class}/{structural_class}_seqs.pt')
    annotated_seqs = []
    for dom in tqdm(all_domains):
        tokens = dom.strip().split('\t')
        if tokens[3].split('.')[0] != structural_class:
            continue
        pdb_id, chain_id = tokens[1], tokens[2][0]
        annotations = tokens[-1].split(',')
        fold = int(annotations[1].split('=')[-1]) 
        sfam = int(annotations[2].split('=')[-1])
        fam = int(annotations[3].split('=')[-1])

        if pdb_id not in all_available_seqs:
            if retry:
                print(f'Retrying {pdb_id} ...')
                _, seq = fetch_single_pid(pdb_id)
                if seq is not None:
                    all_available_seqs[pdb_id] = seq
                else:
                    continue
            else:
                continue
        for entries in all_available_seqs[pdb_id]:
            if chain_id in entries.description.split('|')[-1]:
                annotated_seqs.append((entries.seq, fold, sfam, fam))

    torch.save(all_available_seqs, f'{scope_dir}/class_{structural_class}/{structural_class}_seqs.pt')
    torch.save(annotated_seqs, f'{scope_dir}/class_{structural_class}/annotated_{structural_class}_seqs.pt')

def organize_scope(structural_class='a'):
    annotated_seqs = torch.load(f'{scope_dir}/class_{structural_class}/annotated_{structural_class}_seqs.pt')
    annotated_seqs = [(seq, fo, sf, fa) for seq, fo, sf, fa in annotated_seqs if (len(seq) <= 1022) and (len(seq) >= 100)]
    scope_data = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    fo_size = defaultdict(lambda: 0)
    sf_size = defaultdict(lambda: 0)
    fa_size = defaultdict(lambda: 0)
    for seq, fo, sf, fa in annotated_seqs:
        fo_size[fo] += 1
        sf_size[sf] += 1
        fa_size[fa] += 1
    
    for seq, fo, sf, fa in annotated_seqs:
        if sf_size[sf] > 1:
            scope_data[fo][sf][fa].add(str(seq))
    
    for fo in scope_data.keys():
        for sf in scope_data[fo].keys():
            for fa in scope_data[fo][sf].keys():
                scope_data[fo][sf][fa] = list(scope_data[fo][sf][fa])         
            scope_data[fo][sf] = dict(scope_data[fo][sf])
        scope_data[fo] = dict(scope_data[fo])

    torch.save(dict(scope_data), f'{scope_dir}/class_{structural_class}/filtered_scope_{structural_class}_data.pt')

def create_train_test_split_scope(structural_class='a'):
    set_seed(2603)
    scope_data = torch.load(f'{scope_dir}/class_{structural_class}/filtered_scope_{structural_class}_data.pt')
    all_folds = torch.tensor(list(scope_data.keys()))
    num_fold = all_folds.shape[0]
    num_test_fold = int(num_fold * 0.05)
    shuffle_idx = torch.randperm(num_fold)
    test_fold, train_fold = all_folds[shuffle_idx[:num_test_fold]].numpy(), all_folds[shuffle_idx[num_test_fold:]].numpy()
    fold_n_seq = defaultdict(lambda: 0)
    
    for fo in scope_data.keys():
        for sf in scope_data[fo].keys():
            for fa in scope_data[fo][sf].keys():
                fold_n_seq[fo] += len(scope_data[fo][sf][fa])
    fold_n_seq = dict(fold_n_seq)

    # Build Test Data
    test_seqs = 0
    test_scope = {
        'seq_dict': {},
        'seq_list': []
    }
    for fo in test_fold:
        test_seqs += fold_n_seq[fo]
        test_scope['seq_dict'].update(scope_data[fo])
        for sf in scope_data[fo].keys():
            for fa in scope_data[fo][sf].keys():
                test_scope['seq_list'] += [(sf, fa, i) for i, _ in enumerate(scope_data[fo][sf][fa])]
    print(f'Test Seqs = {test_seqs}')
    
    # Build Train Data
    train_seqs = 0
    train_scope = {
        'seq_dict': {},
        'seq_list': []
    }
    for fo in train_fold:
        train_seqs += fold_n_seq[fo]     
        train_scope['seq_dict'].update(scope_data[fo])   
        for sf in scope_data[fo].keys():
            for fa in scope_data[fo][sf].keys():
                train_scope['seq_list'] += [(sf, fa, i) for i, _ in enumerate(scope_data[fo][sf][fa])]
    print(f'Train Seqs = {train_seqs}')
    torch.save(train_scope, f'{scope_dir}/class_{structural_class}/train_scope_data_{structural_class}.pt')
    torch.save(test_scope, f'{scope_dir}/class_{structural_class}/test_scope_data_{structural_class}.pt')

def merge_train_test_split():
    structural_classes = ['a', 'b', 'c']
    train_scope = {
        'seq_dict': {},
        'seq_list': []
    }
    test_scope = {
        'seq_dict': {},
        'seq_list': []
    }
    for cls in structural_classes:
        cls_train = torch.load(f'{scope_dir}/class_{cls}/train_scope_data_{cls}.pt')
        train_scope['seq_dict'].update(cls_train['seq_dict'])
        train_scope['seq_list'] += cls_train['seq_list']

        cls_test = torch.load(f'{scope_dir}/class_{cls}/test_scope_data_{cls}.pt')
        test_scope['seq_dict'].update(cls_test['seq_dict'])
        test_scope['seq_list'] += cls_test['seq_list']

    torch.save(train_scope, f'{scope_dir}/class_all/train_scope_data_all.pt')
    torch.save(test_scope, f'{scope_dir}/class_all/test_scope_data_all.pt')

def print_scope_statistics(structural_class='a', fold='train'):
    train_scope = torch.load(f'{scope_dir}/class_{structural_class}/{fold}_scope_data_{structural_class}.pt')
    len_buckets = np.zeros(10, dtype=int)
    for sf, fa, j in train_scope['seq_list']:
        bucket = len(train_scope['seq_dict'][sf][fa][j]) // 100 - 1
        len_buckets[bucket] += 1
    print(len_buckets)

def build_adj_list(structural_class='a', fold='train'):
    raw_data = torch.load(f'{scope_dir}/class_{structural_class}/{fold}_scope_data_{structural_class}.pt')
    n_seq = len(raw_data['seq_list'])
    adj_list = [[] for _ in range(n_seq)]
    for u in trange(n_seq - 1):
        sfu, _, _ = raw_data['seq_list'][u]
        for v in range(u + 1, n_seq):
            sfv, _, _ = raw_data['seq_list'][v]
            if sfu == sfv:
                adj_list[u].append(v)
                adj_list[v].append(u)
    torch.save(adj_list, f'{scope_dir}/class_{structural_class}/{fold}_scope_{structural_class}_adjlist.pt')

if __name__ == '__main__':
    download_scope('a')
    download_scope('b')
    download_scope('c')
    annotate_scope('a')
    annotate_scope('b')
    annotate_scope('c')
    organize_scope('a')
    organize_scope('b')
    organize_scope('c')
    create_train_test_split_scope('a')
    create_train_test_split_scope('b')
    create_train_test_split_scope('c')
    merge_train_test_split()
    build_adj_list('all', 'train')
    build_adj_list('all', 'test')