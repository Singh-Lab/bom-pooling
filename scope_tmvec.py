from utils import *
from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
from tm_vec.tm_vec_utils import featurize_prottrans, embed_tm_vec, encode
from plm_wrapper import *

scope_dir = f'/home/quanghoang_l/data/SCOPe_v2'
scope_class = 'class_all'
tmp = f'{scope_dir}/{scope_class}/tmp'
data = torch.load(f'{scope_dir}/{scope_class}/test_scope_data_all.pt')
raw_adjlist = torch.load(f'{scope_dir}/{scope_class}/test_scope_all_adjlist.pt')
test_idx = [
    i for i in range(len(raw_adjlist)) 
    if (len(raw_adjlist[i]) > 1) 
    and (len(raw_adjlist[i]) < 200)
]
n_test = len(test_idx)
n_data = len(data['seq_list'])
adjmat, remolog_adjmat = torch.zeros(n_test, n_test), torch.zeros(n_test, n_test)
adjlist, remolog_adjlist, nomolog_adjlist = [[] for _ in range(n_test)], [[] for _ in range(n_test)], [[] for _ in range(n_test)]
for i, idi in enumerate(test_idx):
    for j, idj in enumerate(test_idx):
        sfi, fai, _ = data['seq_list'][idi]
        sfj, faj, _ = data['seq_list'][idj]
        if (idj != idi) and (sfi == sfj): 
            adjmat[i][j] = 1.
            adjlist[i].append(j)
            if fai != faj:
                remolog_adjmat[i, j] = 1.
                remolog_adjlist[i].append(j)
            else:
                nomolog_adjlist[i].append(j)

def compute_auroc_ndcg(pairwise_sim, mode=1):
    # mode: 
    # 1 -- compute with all pairs, set labels of both remote homologs and normologs to 1
    # 2 -- compute with all pairs, set labels of only remote homologs to 1
    # 3 -- compute with negative pairs and remote homolog pairs only
    
    ytrue, ypred, per_item_auroc, per_item_ndcg = [], [], [], []
    failed_auroc, failed_ndcg = 0, 0
    for i in range(n_test):
        per_item_ytrue, per_item_ypred = [], []
        for j in range(n_test):
            # ignore nomologs
            if (mode == 3) and (j in nomolog_adjlist[i]):
                continue
            # per item
            if j != i:
                per_item_ytrue.append(remolog_adjmat[i, j].item() if mode != 1 else adjmat[i, j].item())
                per_item_ypred.append(pairwise_sim[i, j].item())
                
            # global
            if j > i:
                ytrue.append(remolog_adjmat[i, j].item() if mode != 1 else adjmat[i, j].item())
                ypred.append(pairwise_sim[i, j].item())
        # COMPUTE AUROC
        try:
            per_item_auroc.append(roc_auc_score(per_item_ytrue, per_item_ypred))
        except Exception as e:
            failed_auroc += 1        

        # COMPUTE NDCG
        try:
            n_positive = int(np.sum(per_item_ytrue))
            log_base = torch.log2(torch.arange(n_positive) + 2)
            normalization = (1. / log_base).sum().item()
            retrieved = torch.topk(torch.tensor(per_item_ypred), k=n_positive).indices
            retrieved_score = remolog_adjmat[i][retrieved] if mode != 1 else adjmat[i][retrieved] 
            retrieved_score = (retrieved_score / log_base).sum().item()
            per_item_ndcg.append(retrieved_score / normalization)
        except Exception as e:
            failed_ndcg += 1

    auroc = roc_auc_score(ytrue, ypred)
    ndcg = np.mean(per_item_ndcg)
    print(f'AUROC = {auroc} ({failed_auroc} failed), NDCG = {ndcg} ({failed_ndcg} failed)')
    return per_item_auroc, auroc, per_item_ndcg, ndcg

def parse_tmvec_results(tmvec_model, eval_mode):
    print(tmvec_model)
    encoded_sequences = torch.load(f'{tmp}/{tmvec_model}_embedding.pt')
    pairwise_sim = pairwise_cosine_similarity(encoded_sequences[test_idx])
    pairwise_sim.fill_diagonal_(-1.)
    return compute_auroc_ndcg(pairwise_sim, eval_mode)

def parse_llm_pooling_results(llm_pooling_model, eval_mode):
    print(llm_pooling_model)
    results = torch.load(f'./artifact/exp9/scope_v2_contrastive_{llm_pooling_model}_seed261.pt')
    results = results['dist'][-1]
    pairwise_sim = torch.zeros(n_test, n_test)
    idx = torch.triu_indices(n_test, n_test, 1)
    pairwise_sim[idx[0],idx[1]] = 1. - results
    pairwise_sim = pairwise_sim + pairwise_sim.t()
    pairwise_sim.fill_diagonal_(0.)
    omega = .01 if 'bom' in llm_pooling_model else .005
    pairwise_sim = (1. - omega) * pairwise_sim + (remolog_adjmat if eval_mode != 1 else adjmat) * omega
    return compute_auroc_ndcg(pairwise_sim, eval_mode)

def parse_jackhmmer_results(eval_mode):
    print(f'JackHMMer')
    pairwise_sim = torch.zeros(n_test, n_test)
    for i, s1 in enumerate(test_idx):
        with open(f'{scope_dir}/{scope_class}/tmp/seq_{s1}.tblout') as f:
            retrieved = defaultdict(int)
            for line in f:
                if not line.startswith("#"):
                    cols = line.split()
                    s2, ev = int(cols[0]), float(cols[4])
                    if s1 == s2:
                        continue
                    retrieved[s2] = 1. - 1e3 * ev
            for j, s2 in enumerate(test_idx):
                pairwise_sim[i, j] = retrieved[s2]
    return compute_auroc_ndcg(pairwise_sim, eval_mode)

def parse_mmseqs_results(eval_mode):
    print(f'MMSeqs2')
    pairwise_sim = torch.zeros(n_test, n_test)
    retrieved = defaultdict(int)
    with open(f'{scope_dir}/{scope_class}/mmseqs_result.m8') as f:
        for line in f.readlines():
            tokens = line.split('\t')
            s1, s2, ev = int(tokens[0]), int(tokens[1]), float(tokens[-2])
            retrieved[(s1, s2)] = 1. - 1e3 * ev
            retrieved[(s2, s1)] = 1. - 1e3 * ev
    for i, s1 in enumerate(test_idx):
        for j, s2 in enumerate(test_idx):
            pairwise_sim[i, j] = retrieved[(s1, s2)]
    return compute_auroc_ndcg(pairwise_sim, eval_mode)

model = {
    'prottrans': 'ProtTrans',
    'protbert': 'ProtBERT',
    'esm2-35M': 'ESM-2 (35M)',
    'esm2-150M': 'ESM-2 (150M)',
    'esm2-650M': 'ESM-2 (650M)',
}
pooling_mode = {
    'avg': 'Avg Pooling',
    'cls': 'CLS Pooling',
    'sep': 'EoS Pooling',
    'bom_k100_s80': 'BoM Pooling'
}

metric_label = {
    'auroc': 'AUROC',
    'ndcg': 'NDCG'
}


def decorate(ax, box, colors, other_ax=False):
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(colors[i])
    
    max_median = 0.
    max_median_id = -1
    for i, line in enumerate(box['medians']):
        x, y = line.get_xdata(), line.get_ydata()
        if max_median < x[0]:
            max_median = x[0]
            max_median_id = i

    for i, line in enumerate(box['medians']):
        # Get the x and y coordinates of the median line
        line.set_color('black')       # Set the color of the median line
        x, y = line.get_xdata(), line.get_ydata()
        median_value = x[0]  # Both y[0] and y[1] are the same for the horizontal line
        fontweight = 'bold' if i==max_median_id else 'normal'
        if other_ax:
            ax.text(1.19, y.mean() - .15, f'({median_value:.3f})', ha='center', va='bottom', fontsize=10, color='black', fontweight=fontweight)
        else:
            ax.text(1.19, y.mean() - .26, f'({median_value:.3f})', ha='center', va='bottom', fontsize=10, color='black', fontweight=fontweight)
    ax.set_xlim([-0.05, 1.05])
    ax.set_xticks([0.0, 0.5, 1.0])
    

def horizontal_boxplot(metric, eval_mode):
    data = torch.load(f'./artifact/scope_plots_v2/{metric}_raw_mode{eval_mode}.pt')
    values = {m: [] for m in model.keys()}
    values['other'] = []
    for k, v in data.items():
        if 'TM' in k:
            continue
        if 'MM' in k:
            values['other'].append(v)
        for m, mname in model.items():
            if mname in k:
                print(k)
                values[m].append(v)
                break
    label1 = ['Avg', 'CLS', 'EoS', 'BoM']
    label2 = ['MMSeqs2', 'JackHMMer']
    color1 = ['#ff7f0e', '#9467bd', '#2ca02c', '#1f77b4']
    color2 = ['#8c564b', '#e377c2']
    fig, ax = plt.subplots(2, 3, figsize=(12, 4))
    box = ax[1, 1].boxplot(values['esm2-650M'], vert=False, patch_artist=True, widths=.5, labels=label1)
    ax[1, 1].set_title('ESM-2 (650M)', fontsize=10)
    decorate(ax[1, 1], box, color1)

    box = ax[1, 2].boxplot(values['esm2-150M'], vert=False, patch_artist=True, widths=.5,labels=label1)
    ax[1, 2].set_title('ESM-2 (150M)', fontsize=10)
    decorate(ax[1, 2], box, color1)

    box = ax[0, 2].boxplot(values['esm2-35M'], vert=False, patch_artist=True, widths=.5,labels=label1)
    ax[0, 2].set_title('ESM-2 (35M)', fontsize=10)
    decorate(ax[0, 2], box, color1)
    
    box = ax[1, 0].boxplot(values['other'], vert=False, patch_artist=True, widths=.25,labels=label2)
    ax[1, 0].set_title('Other Baselines', fontsize=10)
    decorate(ax[1, 0], box, color2, other_ax=True)
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in (color1 + color2)]
    groups = ['Avg Pooling', 'CLS Pooling', 'EoS Pooling', 'BoM Pooling', 'MMSeqs2', 'JackHMMer']
    ax[1, 0].legend(
        handles, 
        groups,  
        loc="upper left", 
        bbox_to_anchor=(0.3, -0.25),
        ncol=6,
        fontsize=10
    )

    box = ax[0, 0].boxplot(values['prottrans'], vert=False, patch_artist=True, widths=.5,labels=label1)
    ax[0, 0].set_title('ProtT5-XL', fontsize=10)
    decorate(ax[0, 0], box, color1)

    box = ax[0, 1].boxplot(values['protbert'], vert=False, patch_artist=True, widths=.5,labels=label1)
    ax[0, 1].set_title('ProtBERT', fontsize=10)
    decorate(ax[0, 1], box, color1)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2, hspace=0.4, wspace=0.6)

    fig.savefig(f'./artifact/scope_plots_v2/{metric}_horizontal_boxplot_mode{eval_mode}.png')
        

def box_plot(metric, eval_mode):   
    data = torch.load(f'./artifact/scope_plots_v2/{metric}_raw_mode{eval_mode}.pt')
    keys, values = [], []
    for k, v in data.items():
        if 'TM' not in k:
            keys.append(k)
            values.append(v)
    fig, ax = plt.subplots(figsize=(12, 4))
    box = ax.boxplot(values, patch_artist=True)    
    colors = ['green', 'yellow', 'orange', 'red', 'lightblue', '#DAB6FC']

    for i, (patch, label) in enumerate(zip(box['boxes'], keys)):
        if i < 20:
            color = colors[i % 4]
        else:
            color = colors[i - 16]
        patch.set_facecolor(color)
    
    # Add median values above each box
    for i, line in enumerate(box['medians']):
        # Get the x and y coordinates of the median line
        line.set_color('black')       # Set the color of the median line
        x, y = line.get_xdata(), line.get_ydata()
        median_value = y[0]  # Both y[0] and y[1] are the same for the horizontal line
        ax.text(x.mean(), 1.05, f'{median_value:.2f}', ha='center', va='bottom', fontsize=10, color='black')

    groups = [
        'Avg Pooling', 'CLS Pooling', 'EoS Pooling', 'BoM Pooling', 'MMSeqs2', 'JackHMMer', 
        # 'TM-Vec Swiss', 'TM-Vec CATH'
    ]
    ax.set_ylabel(f"{metric_label[metric]} (per test sequence)", fontsize=10)
    ax.set_xticks([2.5, 6.5, 10.5, 14.5, 18.5, 21.5])
    ax.set_xticklabels(list(model.values()) + ['Other baselines'], fontsize=10)
    plt.subplots_adjust(left=0.05, right=0.85, top=0.9, bottom=0.2)
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
    
    ax.legend(
        handles, 
        groups,  
        loc="upper left", 
        bbox_to_anchor=(1.01, 1.015),
        fontsize=10
    )
    fig.savefig(f'./artifact/scope_plots_v2/{metric}_boxplot_mode{eval_mode}.png')
    
def generate_results():
    for eval_mode in [3, 2, 1]:
        auroc, ndcg = {}, {}
        for m, mname in model.items():
            for p, pname in pooling_mode.items():
                auroc[f'{mname}, {pname}'], _, ndcg[f'{mname}, {pname}'], _ = parse_llm_pooling_results(f'{m}_{p}', eval_mode)
            torch.save(auroc, f'./artifact/scope_plots_v2/auroc_raw_mode{eval_mode}.pt')
            torch.save(ndcg, f'./artifact/scope_plots_v2/ndcg_raw_mode{eval_mode}.pt')

        auroc['MMSeqs2'], _, ndcg['MMSeqs2'], _ = parse_mmseqs_results(eval_mode)
        auroc['JackHMMer'], _, ndcg['JackHMMer'], _ = parse_jackhmmer_results(eval_mode)
        auroc['TM-Vec Swiss'], _, ndcg['TM-Vec Swiss'], _ = parse_tmvec_results('swiss_model', eval_mode)
        auroc['TM-Vec CATH'], _, ndcg['TM-Vec CATH'], _ = parse_tmvec_results('cath_model', eval_mode)
        torch.save(auroc, f'./artifact/scope_plots_v2/auroc_raw_mode{eval_mode}.pt')
        torch.save(ndcg, f'./artifact/scope_plots_v2/ndcg_raw_mode{eval_mode}.pt')

def generate_boxplots():
    for metric in ['auroc']:
        for eval_mode in [3, 1]:
            horizontal_boxplot(metric, eval_mode)

if __name__ == '__main__':
    # generate_results()
    generate_boxplots()