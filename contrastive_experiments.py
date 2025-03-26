from models import *
from contrastive import *

ARTIFACT_DIR = {
    'dpi': f'./artifact/exp1',
    'scope': f'./artifact/exp2',
}

for dir in ARTIFACT_DIR.values(): 
    os.makedirs(dir, exist_ok=True)

def run_contrastive(**kwargs):
    set_seed(kwargs["seed"])
    save_dir = f'{ARTIFACT_DIR[kwargs["dataset"]]}/{kwargs["dataset"]}_contrastive_{kwargs["emb_model"]}_{kwargs["pooling_mode"]}'
    if kwargs['pooling_mode'] == 'bom':
        save_dir += f'_k{kwargs["k"]}_s{kwargs["stride"]}' 
    save_dir += f'_seed{kwargs["seed"]}.pt'
    dist_net_config = {
        'qnet': MLP([PLM_dim[kwargs['emb_model']], 256, 1024]),
        'knet': MLP([PLM_dim[kwargs['emb_model']], 256, 1024]),
        'vnet': MLP([PLM_dim[kwargs['emb_model']], 256, 1024])
    }
    dist_net = CrossAttentionKernel(**dist_net_config) if kwargs['pooling_mode'] == 'bom' else MultiheadLinearKernel(**dist_net_config) 
    if kwargs['dataset'] == 'dpi':
        data = DPIDataset(
            emb_model=kwargs['emb_model'],
            save='load_pool',
            pooling_mode=kwargs['pooling_mode'],
            k=kwargs.get('k', None),
            stride=kwargs.get('stride', None)
        ) 
        model = DPI(
            data=data,
            contrastive_loss=TripletLoss(dist_net, margin=kwargs.get('margin', 1.)),
        )
    if kwargs['dataset'] == 'scope':
        data = SCOPeDataset(
            emb_model=kwargs['emb_model'],
            save='load_pool',
            pooling_mode=kwargs['pooling_mode'],
            k=kwargs.get('k', None),
            stride=kwargs.get('stride', None)
        ) 
        model = SCOPe(
            data=data,
            contrastive_loss=TripletLoss(dist_net, margin=kwargs.get('margin', 1.)),
        )
    model.reset_history()
    model.train(
        n_epochs=kwargs.get('n_epochs', 401),
        interval=kwargs.get('interval', 20),
        batch_size=kwargs.get('batch_size', 128),
        lr=kwargs.get('lr', 4e-4)
    )
    if kwargs['save']:
        torch.save(model.history, save_dir)

    return model

def dpi_experiments(dev):
    torch.cuda.set_device(dev)
    run_experiments(
        common_kwargs = {
            'dataset': 'dpi',
            'save': True,
            'k': 40,
            'stride': 8,
            'n_epochs': 2001,
            'interval': 100,
            'margin': 0.6
        },
        lr={'contrastive': 1e-4},
    )

def scope_experiments(dev):
    torch.cuda.set_device(dev)
    run_experiments(
        common_kwargs = {
            'dataset': 'scope',
            'save': True,
            'k': 100,
            'stride': 20,
            'n_epochs': 201,
            'interval': 10,
            'list_only': list_only
        },
        lr={'contrastive': 5e-4},
    )

def run_experiments(common_kwargs, lr):
    pooling_method = [
        'bom', 
        'cls', 
        'avg',
        'sep'
    ]
    emb_models = [
        'prottrans', 
        'protbert',
        'esm2-35M', 
        'esm2-150M',
        'esm2-650M'
    ]
    seeds = [
        261,
        2602,
        26003,
        2604,
        265
    ]
    for seed in seeds:
        common_kwargs['seed'] = seed
        for pm in pooling_method:
            for em in emb_models:           
                run_contrastive(**common_kwargs, pooling_mode=pm, emb_model=em, lr=lr['contrastive'])

if __name__ == '__main__':
    dpi_experiments(dev=0)