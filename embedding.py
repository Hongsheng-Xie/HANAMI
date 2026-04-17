import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pubchempy as pcp
import time
import gc
from Bio import Entrez, SeqIO

from transformers import AutoTokenizer, AutoModel
from chemprop.models.model import MPNN
from chemprop.data import MoleculeDataset
from chemprop.nn import BondMessagePassing, NormAggregation, RegressionFFN
from chemprop.nn import metrics
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader

from enformer_pytorch import Enformer
from borzoi_pytorch import Borzoi  
from sklearn.decomposition import PCA

#ChemBERTa
df = pd.read_csv('./data/drkg/drug_SMILES.csv')
smile_list = df["drug_SMILES"].tolist()

tokenizer1 = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model1 = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

inputs = tokenizer1(smile_list, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model1(**inputs)
    feature_matrix1 = outputs.last_hidden_state[:, 0, :].numpy()
matrix_tensor1 = torch.tensor(feature_matrix1)

tokenizer2 = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
model2 = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

inputs = tokenizer2(smile_list, return_tensors="pt", padding=True, truncation=True)


with torch.no_grad():
    outputs = model2(**inputs)
    feature_matrix2 = outputs.last_hidden_state[:, 0, :].numpy()
matrix_tensor2 = torch.tensor(feature_matrix2)

drug_feat = torch.cat((matrix_tensor1,matrix_tensor2), 1)

#MPNN
mp = BondMessagePassing()
agg = NormAggregation()
ffn = RegressionFFN()

df_drkg = pd.read_csv('./data/drkg/drug_SMILES.csv')

smiles_list = df_drkg["drug_SMILES"].tolist()
basic_model = MPNN(mp, agg, ffn)

ys = np.random.rand(len(smiles_list), 1)

dataset = MoleculeDataset([MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smiles_list, ys)])
dataloader = build_dataloader(dataset, shuffle=False)

basic_model.eval()
all_embeddings = []

with torch.no_grad():
    for batch in dataloader:
        bmg, V_d, X_d, *_ = batch
        embeddings = basic_model.fingerprint(bmg, V_d, X_d)  # [batch_size, embedding_dim]
        all_embeddings.append(embeddings.cpu())
    
feature_matrix = torch.cat(all_embeddings, dim=0).numpy()
matrix_tensor = torch.tensor(feature_matrix)



ys = np.random.rand(len(smiles_list), 1)
dataset = MoleculeDataset([MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smiles_list, ys)])
dataloader = build_dataloader(dataset,shuffle=False)

basic_model.eval()

all_embeddings = []
with torch.no_grad():
    for batch in dataloader:
        bmg, V_d, X_d, *_ = batch
        embeddings = basic_model.fingerprint(bmg, V_d, X_d)  # [batch_size, embedding_dim]
        all_embeddings.append(embeddings.cpu())
    
feature_matrix = torch.cat(all_embeddings, dim=0).numpy()

matrix_tensor = torch.tensor(feature_matrix)

#BioBERT & ClinicalBERT
df_drkg = pd.read_excel('./data/drkg/dise_MeSH.xlsx')

d = df_drkg['Disease'].copy()

tokenizer1 = AutoTokenizer.from_pretrained("dmis-lab/biobert-large-cased-v1.1-squad")
model1 = AutoModel.from_pretrained("dmis-lab/biobert-large-cased-v1.1-squad")
# Get embeddings for each disease name
embeddings_1 = []
for disease in d:
    inputs = tokenizer1(disease, return_tensors="pt", truncation=True, max_length=32)
    with torch.no_grad():
        outputs = model1(**inputs)
        # Use the [CLS] token embedding as a sentence-level embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings_1.append(cls_embedding)

dise_feat1 = torch.tensor(embeddings_1)

tokenizer2 = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model2 = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

embeddings_2 = []

for disease in d:
    inputs = tokenizer2(disease, return_tensors="pt", truncation=True, max_length=32)
    with torch.no_grad():
        outputs = model2(**inputs)
        # Use the [CLS] token embedding as a sentence-level embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings_2.append(cls_embedding)

dise_feat2 = torch.tensor(embeddings_2)

dise_feat = torch.cat((dise_feat1,dise_feat2), 1)

# Borzoi
class FourLayerCNNEncoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super(FourLayerCNNEncoder, self).__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=2, padding=2),  
            nn.ReLU()
        )
        
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(159848*2, 8192),
            nn.ReLU(),
            nn.Linear(8192, 1024)
        )

        # Decoder: transforms back from bottleneck to input_dim
        self.decoder_fc = nn.Sequential(
            nn.Linear(1024, 8192),
            nn.ReLU(),
            nn.Linear(8192, 159848*2),
            nn.ReLU()
        )
        
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (8,53*29,2*13)),
            nn.ConvTranspose2d(8, 4, kernel_size=4,stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # output normalized between 0 and 1
        )
    def forward(self, x):
        x = self.encoder_conv(x)
        x = self.encoder_fc(x)
        latent = x  # Encoded latent vector
        x = self.decoder_fc(latent)
        x = self.decoder_conv(x)
        return x, latent
    
# create a one-hot encoded DNA sequence, [batch, bp, 4]
def one_hot_encode_dna(seq):
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for i, base in enumerate(seq):
        if base in mapping:
            arr[i, mapping[base]] = 1.0
    return arr

sequence = [line.strip() for line in open('gene_seq.txt', 'r')] #gene_seq.txt contains the gene sequences mapped from NCBI ids.
l = len(sequence)
short = []
m = 524288
s = 0
for i in range(l):
    seq = sequence[i]
    if len(seq) < 524288:
        short.append(i)
        s+=1
        if len(seq) < m:
            m = len(seq)

for i in short:
    seq = sequence[i]
    l = len(seq)
    sub = (524288 - l) // 2
    if sub*2+l == 524288:
        sequence[i] = "N"*sub+seq+"N"*sub
    else:
        sequence[i] = "N"*sub+seq+"N"*(sub+1)
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Borzoi.from_pretrained('johahi/borzoi-replicate-0')
model.to(device)
model.eval()  # Switch to evaluation mode

total_feat_mat = [] 
for idx, item in enumerate(sequence):
    start = time.time()
    # Do your work here (limit to work_time duration)

    # One-hot encode and add batch dimension
    input_tensor = one_hot_encode_dna(item)  # Shape: [1, 524288, 4]
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)
    input_tensor = input_tensor.permute(0,2,1)  # [1, 4, 524288]

    with torch.no_grad():
        embeddings = model(input_tensor)  # [1, n_bins (~4096), embedding_dim (~1536)]
        embeddings = embeddings[0]
        X_t = embeddings.T

        pca = PCA(n_components=100)
        X_reduced = pca.fit_transform(X_t)
        
        reduced = X_reduced.T
        
        # Convert to tensor if needed
        if not isinstance(reduced, torch.Tensor):
            reduced_tensor = torch.tensor(reduced, dtype=torch.float32)
        else:
            reduced_tensor = reduced.float()
            
        # Add batch and channel dims: (batch_size=1, channels=1, H, W)
        input_tensor = reduced_tensor.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)
        
        CNN = FourLayerCNNEncoder()
        CNN.eval()
    
        with torch.no_grad():
            _, latent_vectors = CNN(input_tensor)
            
        encoded_list = latent_vectors.squeeze(0).cpu().numpy().tolist()
        
    total_feat_mat.append(encoded_list) #np.concatenate((total_feat_mat, gene_embedding), axis=0)
        
    del embeddings,encoded_list,latent_vectors
    #del gene_embedding
    if idx % 10 == 0:
        time.sleep(1)
        gc.collect()

total_feat_mat = np.array(total_feat_mat)
tensor = torch.from_numpy(total_feat_mat)


# Enformer
class FourLayerCNNEncoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super(FourLayerCNNEncoder, self).__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=2),  
            nn.ReLU()
        )
        
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*224*25, 8192),
            nn.ReLU(),
            nn.Linear(8192, 1024)
        )

        # Decoder: transforms back from bottleneck to input_dim
        self.decoder_fc = nn.Sequential(
            nn.Linear(1024, 8192),
            nn.ReLU(),
            nn.Linear(8192, 32*224*25),
            nn.ReLU()
        )
        
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (32,224,25)),
            nn.ConvTranspose2d(32, 16, kernel_size=4,stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # output normalized between 0 and 1
        )
    def forward(self, x):
        x = self.encoder_conv(x)
        x = self.encoder_fc(x)
        latent = x  # Encoded latent vector
        x = self.decoder_fc(latent)
        x = self.decoder_conv(x)
        return x, latent
    
df_gene = np.load('./data/ms/id2gene.npy', allow_pickle=True)

no_id_gene = df_gene.item()
NCBI_gene = no_id_gene.values()
NCBI_gene = list(NCBI_gene)

#convert NCBI id to gene name

h = 0
m = 0
P_gene = []
total_feat_mat = [] 
for i in range(len(NCBI_gene)):
    gene_id = NCBI_gene[i]
    Entrez.email = "xxx@xxx.com" # xxx@xxx.com use personal email address signed up in Entrez 
    handle = Entrez.esummary(db="gene", id=gene_id, retmode="xml")
    record = Entrez.read(handle)
    gen_info = record['DocumentSummarySet']['DocumentSummary'][0]['GenomicInfo'][0]
    accession = gen_info['ChrAccVer']

    doc_summary = record['DocumentSummarySet']['DocumentSummary'][0]
    organism = doc_summary['Organism']

    start = int(gen_info['ChrStart']) + 1  # Adjust for 1-based indexing
    end = int(gen_info['ChrStop']) + 1
    midpoint = (start + end) // 2
    #print(midpoint)
    start = midpoint - 98303
    end = midpoint + 98304
    handle = Entrez.efetch(
        db="nucleotide",
        id=accession,
        seq_start=start,
        seq_stop=end,
        rettype="fasta",
        retmode="text"
    )
    
    record = SeqIO.read(handle, "fasta")

    #Enformer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Enformer.from_pretrained('EleutherAI/enformer-official-rough')
    model.to(device)
    model.eval()
    #seq = torch.randint(0, 4, (1, 196608))
    seq = record.seq
    nt_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    int_encoded = [nt_to_int.get(base, 4) for base in seq]

    sequence = torch.tensor(int_encoded)
    with torch.no_grad():
        output = model(sequence)  # output is a dict: {'human': 5313, 'mouse': 1643}

        if organism['CommonName'] == 'human': 
            tracks = output['human']  # shape: (batch, 896, 5313)
        elif organism['CommonName'] == 'house mouse':
            tracks = output['mouse']

        tracks = tracks.cpu().detach().numpy()  # If it's a torch tensor

        # Apply PCA to reduce to 100 principal components
        pca = PCA(n_components=100)
        X_reduced = pca.fit_transform(tracks)
            
        reduced = X_reduced.T
        
        # Convert to tensor if needed
        if not isinstance(reduced, torch.Tensor):
            reduced_tensor = torch.tensor(reduced, dtype=torch.float32)
        else:
            reduced_tensor = reduced.float()
            
        # Add batch and channel dims: (batch_size=1, channels=1, H, W)
        input_tensor = reduced_tensor.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)
        
        CNN = FourLayerCNNEncoder()
        CNN.eval()

        with torch.no_grad():
            _, latent_vectors = CNN(input_tensor)
            
        encoded_list = latent_vectors.squeeze(0).cpu().numpy().tolist()
        
    total_feat_mat.append(encoded_list)
    
    P_gene.append(reduced)

    del tracks,encoded_list,latent_vectors
    #del gene_embedding
    if i % 10 == 0:
        time.sleep(1)
        gc.collect()

total_feat_mat = np.array(total_feat_mat)
tensor = torch.from_numpy(total_feat_mat)