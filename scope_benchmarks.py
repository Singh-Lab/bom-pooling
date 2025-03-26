from utils import *
from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
from tm_vec.tm_vec_utils import featurize_prottrans, embed_tm_vec, encode
from plm_wrapper import *

scope_dir = f'/home/quanghoang_l/data/SCOPe_v2'
scope_class = 'class_all'
tmp = f'{scope_dir}/{scope_class}/tmp'

def generate_fasta():
    data = torch.load(f'{scope_dir}/{scope_class}/test_scope_data_all.pt')
    # create fasta
    with open(f'{scope_dir}/{scope_class}/test_scope_data_all.fasta', 'w') as f:
        for i, (sf, fa, id) in enumerate(data['seq_list']):
            seq = data['seq_dict'][sf][fa][id]
            f.write(f'>{i}\n{seq}\n')

def construct_remote_only_adjlist():
    data = torch.load(f'{scope_dir}/{scope_class}/test_scope_data_all.pt')
    adjlist = [[] for _ in range(len(data['seq_list']))]
    for i, (sfi, fai, idi) in enumerate(data['seq_list']):
        for j, (sfj, faj, idj) in enumerate(data['seq_list']):
            if (j != i) and (sfi == sfj) and (fai != faj):
                adjlist[i].append(j)
    return adjlist

def run_mmseqs():
    fasta_file = f'{scope_dir}/{scope_class}/test_scope_data_all.fasta'
    res_file = f'{scope_dir}/{scope_class}/mmseqs_result.m8'
    tmp = f'{scope_dir}/{scope_class}/tmp'
    os.system(f'mmseqs easy-search {fasta_file} {fasta_file} {res_file} {tmp}')

def run_jackhmmer(scope_class):
    fasta_file = f'{scope_dir}/{scope_class}/test_scope_data_all.fasta'
    tmp = f'{scope_dir}/{scope_class}/tmp'
    adjlist = torch.load(f'{scope_dir}/{scope_class}/test_scope_all_adjlist.pt')
    
    test_idx = [
        i for i in range(len(adjlist)) 
        if (len(adjlist[i]) > 1) 
        and (len(adjlist[i]) < 200)
    ]
    data = torch.load(f'{scope_dir}/{scope_class}/test_scope_data_all.pt')
    for i in test_idx:
        sf, fa, id = data['seq_list'][i]
        seq = data['seq_dict'][sf][fa][id]
        with open(f'{tmp}/seq_{i}.fasta', 'w') as f:
            f.write(f'>{i}\n{seq}\n')
        os.system(f'jackhmmer -A {tmp}/seq_{i}.hmmalign --tblout {tmp}/seq_{i}.tblout --noali --cpu 4 -E 1e-3 {tmp}/seq_{i}.fasta {fasta_file}')

def generate_tmvec_embedding(tmvec_model):
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to('cuda')
    gc.collect()

    cache_dir = f'/home/quanghoang_l/.cache/tm-vec'
    #TM-Vec model paths
    tm_vec_model_cpnt = f'{cache_dir}/tm_vec_{tmvec_model}.ckpt'
    tm_vec_model_config = f'{cache_dir}/tm_vec_{tmvec_model}_params.json'

    #Load the TM-Vec model
    tm_vec_model_config = trans_basic_block_Config.from_json(tm_vec_model_config)
    model_deep = trans_basic_block.load_from_checkpoint(tm_vec_model_cpnt, config=tm_vec_model_config)
    model_deep = model_deep.to('cuda')
    model_deep = model_deep.eval()

    # Generate embedding
    fasta_data = SeqIO.parse(open(f'{scope_dir}/{scope_class}/test_scope_data_all.fasta'), 'fasta')
    batch, batch_size = [], 128
    encoded_sequences = []
    for i, fasta in tqdm(enumerate(fasta_data)):
        if len(batch) < batch_size:
            batch.append(fasta.seq)
            continue
        else: 
            with torch.no_grad():
                batch_emb = torch.tensor(encode(batch, model_deep, model, tokenizer, 'cuda'))
                encoded_sequences.append(batch_emb)
            batch = [fasta.seq]
    last_batch_emb = torch.tensor(encode(batch, model_deep, model, tokenizer, 'cuda'))
    encoded_sequences.append(last_batch_emb)
    encoded_sequences = torch.cat(encoded_sequences, dim=0)
    torch.save(encoded_sequences, f'{tmp}/{tmvec_model}_embedding.pt')

if __name__ == '__main__':
    # generate_fasta()
    # run_mmseqs()
    # run_jackhmmer()
    # generate_tmvec_embedding('swiss')
    # generate_tmvec_embedding('cath')