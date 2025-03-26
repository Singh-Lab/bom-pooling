from plm_wrapper import *
from utils import *
from scipy.stats import spearmanr
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import time

ARTIFACT_DIR = {
    'beta_lactamase': f'./artifact/exp3',
    'fluorescence': f'./artifact/exp4',
}

for dir in ARTIFACT_DIR.values(): 
    os.makedirs(dir, exist_ok=True)

def evaluate_model(model, data, fold='test'):
    eval_data = data[f'{fold}_x']
    eval_label = data[f'{fold}_y']
    n_test = len(eval_data)
    batch_size = 128
    bar = trange(0, n_test, batch_size)
    bar.set_description_str(f'Evaluating on {fold} fold:')
    y_pred = []
    for i in bar:
        x_batch = eval_data[i: min(i + batch_size, n_test)].to('cuda')
        y_pred.append(model(x_batch))
        del(x_batch)
        torch.cuda.empty_cache()
    y_pred = torch.cat(y_pred, dim=0).cpu()
    rho, _ = spearmanr(y_pred.numpy(), eval_label.cpu().numpy())
    return y_pred, rho

def train_model(task, emb_model, pooling_mode, seed, **kwargs):
    set_seed(seed)
    exp_name = f'{task}_{emb_model}_{pooling_mode}'
    tokens = pooling_mode.split('_')
    pooling_mode = tokens[0]
    if pooling_mode == 'bom':
        k, s = int(tokens[1][1:]), int(tokens[2][1:])
        trainable = False
    else:
        k, s = 1, 1
        trainable = True if len(tokens) > 1 else 0
    exp_name = exp_name + '_train' if trainable else ''
    data = load_data(task, emb_model, pooling_mode, k=k, s=s)
    model = LLMPredictor(emb_model=emb_model, pooling_mode=pooling_mode, k=k, s=s, trainable=trainable)
    opt = AdamW(model.parameters(), lr=kwargs.get('lr', 1e-4))
    scheduler = ReduceLROnPlateau(opt, 'min', min_lr=1e-6, factor=0.95, patience=1)
    n_train = len(data['train_x'])
    batch_size = kwargs.get('batch_size', 256)
    history = {
        'y_pred': [],
        'rho': [],
        'training_time': []
    }
    for ep in range(kwargs.get('n_epochs', 400)):
        if ep % kwargs.get('interval', 1) == 0:
            with torch.no_grad():
                model.eval()
                y_pred, rho = evaluate_model(model, data)
                _, rho_valid = evaluate_model(model, data, 'valid')
                print(f"Spearman's rho test: {rho} valid: {rho_valid} lr={scheduler.get_last_lr()}") 
                history['y_pred'].append(y_pred)
                history['rho'].append(rho)
                torch.save(history, f'{ARTIFACT_DIR[task]}/{exp_name}_seed{seed}.pt')
                scheduler.step(- rho_valid)
                model.train()
        start_time = time.time()
        idx = torch.randperm(n_train)
        bar = trange(0, n_train, batch_size)
        bar.set_description_str(f'Epoch {ep}:')
        for i in bar:
            opt.zero_grad()
            idx_batch = idx[i: min(i + batch_size, n_train)]
            x_batch = data['train_x'][idx_batch].to('cuda')
            y_batch = data['train_y'][idx_batch].to('cuda')
            y_pred = model(x_batch)
            loss = F.huber_loss(y_pred, y_batch)
            bar.set_postfix_str(f'Batch Loss = {loss.item():.5f}')
            loss.backward()
            opt.step()
            del(x_batch)
        ep_time = time.time() - start_time
        history['training_time'].append(ep_time)
        torch.save(history, f'{ARTIFACT_DIR[task]}/{exp_name}_seed{seed}.pt')
        print(f"Ep {ep} training time: {ep_time}")
    del data, model
    torch.cuda.empty_cache()

tasks = [
    'beta_lactamase',
    'fluorescence'
]
seeds = [
    261,
    2602,
    26003,
    2604,
    265
]
emb_models = [
    'protbert',
    'prottrans',
    'esm2-35M',
    'esm2-150M',
    'esm2-650M'
]    
pooling_mode = [
    'bom_k1_s1',
    'bom_k20_s4',
    'bom_k40_s8',
    'bom_k60_s12',
    'bom_k80_s16',
    'bom_k100_s20',
    'avg',
    'cls',
    'sep',
    'avg_v',
    'cls_v',
    'sep_v'
]

def slurm():
    kwargs = {
        'n_epochs': 101,
        'interval': 100,
        'batch_size': 48,
        'lr': 1e-4
    }

    arguments = [(t, emb, s, p) for t in tasks for s in seeds for emb in emb_models for p in pooling_mode]
    print(len(arguments))
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    t, emb, s, p = arguments[idx]
    train_model(t, emb, p, s, **kwargs)

if __name__ == '__main__':
    slurm()