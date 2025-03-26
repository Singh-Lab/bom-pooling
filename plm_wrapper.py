from utils import *

from transformers import BertModel, BertTokenizer, T5EncoderModel, T5Tokenizer, EsmTokenizer, EsmModel, AutoTokenizer

PROTBERT_MODEL_NAME = 'Rostlab/prot_bert'
PROTTRANS_MODEL_NAME = 'Rostlab/prot_t5_xl_uniref50'
ESM2_35M_MODEL_NAME = 'facebook/esm2_t12_35M_UR50D'
ESM2_150M_MODEL_NAME = 'facebook/esm2_t30_150M_UR50D'
ESM2_650M_MODEL_NAME = 'facebook/esm2_t33_650M_UR50D'
class ProtTransWrapper(nn.Module):
    def __init__(self, freeze=True):
        super(ProtTransWrapper, self).__init__()
        self.prot_trans = T5EncoderModel.from_pretrained(PROTTRANS_MODEL_NAME)
        if freeze:
            freeze_module(self.prot_trans)
        self.tokenizer = T5Tokenizer.from_pretrained(PROTTRANS_MODEL_NAME, do_lower_case=False)

    def tokenize(self, input_seqs):
        tokenized_seqs = self.tokenizer.batch_encode_plus(
            input_seqs, add_special_tokens=True,
            padding="longest", return_tensors='pt'
        )
        for key in tokenized_seqs.keys():
            tokenized_seqs[key] = tokenized_seqs[key].to(self.prot_trans.device)
        return tokenized_seqs

    def forward(self, inputs, raw_seq=True):
        if raw_seq:
            inputs = [' '.join(x) for x in inputs]
            tokenized_seqs = self.tokenize(inputs)
            input_ids, mask = tokenized_seqs['input_ids'], tokenized_seqs['attention_mask']
        else:
            input_ids, mask = inputs[0].to(self.prot_trans.device), inputs[1].to(self.prot_trans.device)
        embedding_repr = self.prot_trans(input_ids, mask)
        return embedding_repr.last_hidden_state


class ProtBERTWrapper(nn.Module):
    def __init__(self, freeze=True):
        super(ProtBERTWrapper, self).__init__()
        self.prot_bert = BertModel.from_pretrained(PROTBERT_MODEL_NAME)
        if freeze:
            freeze_module(self.prot_bert)
        self.tokenizer = BertTokenizer.from_pretrained(PROTBERT_MODEL_NAME, do_lower_case=False)

    def tokenize(self, input_seqs):
        # tokenize input_seqs with ProtBert
        # input_seqs is list of space-delimited strings
        tokenized_seqs = self.tokenizer.batch_encode_plus(
            input_seqs, add_special_tokens=True,
            padding="longest", return_tensors='pt'
        )
        for key in tokenized_seqs.keys():
            tokenized_seqs[key] = tokenized_seqs[key].to(self.prot_bert.device)
        return tokenized_seqs

    def embed(self, inputs):
        return self.forward(inputs, prompt=None)

    def prompted_embed(self, inputs, prompt):
        return self.forward(inputs, prompt=prompt)
 
    def forward(self, inputs, position_ids=None, prompt=None, remove_prompt_embedding=False):
        inputs = [' '.join(x) for x in inputs]
        tokenized_seqs = self.tokenize(inputs)
        input_ids, mask = tokenized_seqs['input_ids'], tokenized_seqs['attention_mask']
        extended_attention_mask = self.prot_bert.get_extended_attention_mask(mask, input_ids)
        head_mask = self.prot_bert.get_head_mask(None, self.prot_bert.config.num_hidden_layers)
        embedding_output = self.prot_bert.embeddings(input_ids=input_ids, position_ids=position_ids)
        hidden_states = embedding_output
        if prompt is not None:
            # expect prompt of shape [n_tokens, 1024]
            hidden_states = torch.cat([
                hidden_states[:, :1, :],    # CLS token
                prompt.repeat(hidden_states.shape[0], 1, 1),    # prompt
                hidden_states[:, 1:, :]     # AA embeddings
            ], dim=1)
            extended_attention_mask = F.pad(extended_attention_mask, (0, prompt.shape[0]))
        for i, layer_module in enumerate(self.prot_bert.encoder.layer):
            layer_head_mask = head_mask[i]
            layer_outputs = layer_module(
                hidden_states,
                extended_attention_mask,
                layer_head_mask
            )
            hidden_states = layer_outputs[0]
        if (prompt is not None) and remove_prompt_embedding:
            hidden_states = torch.cat([
                hidden_states[:, :1, :],    # CLS token
                hidden_states[:, 1 + prompt.shape[0]:, :]   # AA embeddings
            ], dim=1)

        return hidden_states

class ESMWrapper(nn.Module):
    def __init__(self, chkpoint, freeze=True):
        super(ESMWrapper, self).__init__()
        self.esm = EsmModel.from_pretrained(chkpoint)
        if freeze:
            freeze_module(self.esm)
        self.tokenizer = AutoTokenizer.from_pretrained(chkpoint)

    def forward(self, inputs, raw_seq=True):
        if raw_seq:
            tokenized_seqs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
            input_ids, mask = tokenized_seqs['input_ids'].to(self.esm.device), tokenized_seqs['attention_mask'].to(self.esm.device)
        else:
            input_ids, mask = inputs[0].to(self.esm.device), inputs[1].to(self.esm.device)
        embedding_repr = self.esm(input_ids, mask)
        return embedding_repr.last_hidden_state

PLM = {
    'prottrans': lambda: ProtTransWrapper().to('cuda'),
    'protbert': lambda: ProtBERTWrapper().to('cuda'),
    'esm2-35M': lambda: ESMWrapper(ESM2_35M_MODEL_NAME).to('cuda'),
    'esm2-150M': lambda: ESMWrapper(ESM2_150M_MODEL_NAME).to('cuda'),
    'esm2-650M': lambda: ESMWrapper(ESM2_650M_MODEL_NAME).to('cuda'),
}

PLM_dim = {
    'prottrans': 1024,
    'protbert': 1024,
    'esm2-35M': 480,
    'esm2-150M': 640,
    'esm2-650M': 1280
}