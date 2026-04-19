import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from openpyxl import Workbook
import torch
df_drug_drkg = np.load('./data/drkg/id2drug.npy', allow_pickle=True).item()
df_dise_drkg = np.load('./data/drkg/id2dise.npy', allow_pickle=True).item()
df_gene_drkg = np.load('./data/drkg/id2gene.npy', allow_pickle=True).item()

df_drug_ms = np.load('./data/ms/id2drug.npy', allow_pickle=True).item()
df_dise_ms = np.load('./data/ms/id2dise.npy', allow_pickle=True).item()
df_gene_ms = np.load('./data/ms/id2gene.npy', allow_pickle=True).item()

drug_ms = set(df_drug_ms.values())
filtered_drug = {k: v for k, v in df_drug_drkg.items() if v not in drug_ms}

dise_ms = set(df_dise_ms.values())
filtered_dise = {k: v for k, v in df_dise_drkg.items() if v not in dise_ms}

gene_ms = set(df_gene_ms.values())
filtered_gene = {k: v for k, v in df_gene_drkg.items() if v not in gene_ms}


drkg_dise_feat = torch.load('./data/drkg/dise_feat.pth')
drkg_dise = list(filtered_dise.keys())
drkg_dise_feat_selected = drkg_dise_feat[drkg_dise,:]
drkg_dise_feat_selected = drkg_dise_feat_selected[:,drkg_dise]
torch.save(drkg_dise_feat_selected, 'C:/Users/xiehs/Downloads/TriMoGCL-main/TriMoGCL-main/data/drkg/DRKG_MS_dise_Base.pth')

drkg_dise_feat = torch.load('./data/drkg/dise_Bio.pth')
drkg_dise = list(filtered_dise.keys())
drkg_dise_feat_selected = drkg_dise_feat[drkg_dise,:]
torch.save(drkg_dise_feat_selected, 'C:/Users/xiehs/Downloads/TriMoGCL-main/TriMoGCL-main/data/drkg/DRKG_MS_dise_Rev.pth')
l_dise = len(drkg_dise)
dise_dict = {}
for i in range(l_dise):
    dise_dict[drkg_dise[i]] = i
np.save('./data/drkg/subgraph_dise.npy', dise_dict)
#print(drkg_dise_feat_selected.shape) #[1551,1792]

drkg_drug_feat = torch.load('./data/drkg/drug_feat.pth')
drkg_drug = list(filtered_drug.keys())
drkg_drug_feat_selected = drkg_drug_feat[drkg_drug,:]
torch.save(drkg_drug_feat_selected, 'C:/Users/xiehs/Downloads/TriMoGCL-main/TriMoGCL-main/data/drkg/DRKG_MS_drug_Base.pth')
drkg_drug_feat = torch.load('./data/drkg/drug_ALL.pth')
drkg_drug = list(filtered_drug.keys())
drkg_drug_feat_selected = drkg_drug_feat[drkg_drug,:]
torch.save(drkg_drug_feat_selected, 'C:/Users/xiehs/Downloads/TriMoGCL-main/TriMoGCL-main/data/drkg/DRKG_MS_drug_Rev.pth')
l_drug = len(drkg_drug)
drug_dict = {}
for i in range(l_drug):
    drug_dict[drkg_drug[i]] = i
np.save('./data/drkg/subgraph_drug.npy', drug_dict)
#print(drkg_drug_feat_selected.shape) #[1636,1452]

drkg_gene_feat = torch.load('./data/drkg/gene_feat.pth')
drkg_gene = list(filtered_gene.keys())
drkg_gene_feat_selected = drkg_gene_feat[drkg_gene,:]
torch.save(drkg_gene_feat_selected, 'C:/Users/xiehs/Downloads/TriMoGCL-main/TriMoGCL-main/data/drkg/DRKG_MS_gene_Base.pth')
drkg_gene_feat = torch.load('./data/drkg/gene_feat.pth')
drkg_gene = list(filtered_gene.keys())
drkg_gene_feat_selected = drkg_gene_feat[drkg_gene,:]
torch.save(drkg_gene_feat_selected, 'C:/Users/xiehs/Downloads/TriMoGCL-main/TriMoGCL-main/data/drkg/DRKG_MS_gene_Rev.pth')
l_gene = len(drkg_gene)
gene_dict = {}
for i in range(l_gene):
    gene_dict[drkg_gene[i]] = i
np.save('./data/drkg/subgraph_gene.npy', gene_dict)
#print(drkg_gene_feat_selected.shape) #[5291,1024]




