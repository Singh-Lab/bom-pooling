import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pprint import pprint
from torch.optim import AdamW
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import random, os, gc, re, sys, requests
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance
from Bio import SeqIO, pairwise2
from io import StringIO
import glob
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

ARTIFACT_ROOT = f'/home/qh1002/Code/bom_pooling/artifact'

# PyTorch utilities
def freeze_module(module, unfreeze=False):
    for param in module.parameters():
        param.requires_grad = unfreeze

def set_seed(seed=2603):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def inverse_sigmoid(x, temp):
    return torch.log(x / (1. - x)) / temp

def count_params(net):
    return sum(p.numel() for p in net.parameters())

# String processing utilities
def write_fasta(seq_dict, fasta_file, chunk=50):
    with open(fasta_file, 'w') as f:
        for seq_name, seq in seq_dict.items():
            chunk_seq = '\n'.join([seq[i:i+chunk] for i in range(0, len(seq), chunk)])
            f.write(f'>{seq_name}\n')
            f.write(f'{chunk_seq}\n')


def run_clustalw(seq_dict, artifact_folder, fasta_name='temp', msf_name='temp', remove_fasta=False):
    os.makedirs(artifact_folder, exist_ok=True)
    fasta_file = f'{artifact_folder}/{fasta_name}.fasta'
    msf_file = f'{artifact_folder}/{msf_name}.msf'
    write_fasta(seq_dict, fasta_file)
    cline = ClustalwCommandline(
        cmd="/home/quanghoang_l/miniconda3/envs/torch-env/bin/clustalw",
        infile=fasta_file,
        outfile=msf_file,
        output='gcg',
    )
    if remove_fasta:
        os.system(f'rm {fasta_file}')
    return cline()


def extract_kmer(s, k, o=None, plm_format=True):
    o = k - 1 if o is None else o
    kmers = [s[j:j + k] for j in range(0, len(s) - k + 1, k - o)]
    if plm_format:
        return [' '.join(kmer) for kmer in kmers]
    return kmers

def fetch_single_uid(uid):
    base_url="http://www.uniprot.org/uniprot"
    try:
        seq = StringIO(''.join(requests.post(f'{base_url}/{uid}.fasta').text))
        seq = list(SeqIO.parse(seq,'fasta'))
        return (uid, str(seq[0].seq))
    except Exception as e:
        print(f'failed to retrieve uid {uid}')
        return (uid, None)
    
def fetch_single_pid(pid):
    base_url = f"https://www.ebi.ac.uk/pdbe/entry/pdb"
    try:
        seqs = StringIO(''.join(requests.post(f'{base_url}/{pid}/fasta').text))
        seqs = list(SeqIO.parse(seqs,'fasta'))
        if len(seqs) > 0:
            return (pid, seqs)
        else:
            return (pid, None)
    except Exception as e:
        print(e)
        return (pid, None)

def download_uniprot(uid_list):    
    with ThreadPoolExecutor() as executor:
        retrieved_seqs = list(tqdm(executor.map(fetch_single_uid, uid_list), total=len(uid_list)))
    
    return {uid: seq for (uid, seq) in retrieved_seqs if seq is not None}    


def download_pdb(pid_list):
    with ThreadPoolExecutor(max_workers=len(pid_list) // 50) as executor:
        retrieved_seqs = list(tqdm(executor.map(fetch_single_pid, pid_list), total=len(pid_list)))
    
    success = {pid: seq for (pid, seq) in retrieved_seqs if seq is not None}    
    failure = [pid for (pid, seq) in retrieved_seqs if seq is None]
    return success, failure