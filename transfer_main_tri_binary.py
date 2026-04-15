import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os.path
from transfer_util import prepare_data
from base_gcn import GCN_binary_SAGE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,\
precision_recall_curve, auc, f1_score
import pandas as pd
import copy
import sys


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2none(v):
    if v.lower() == 'none':
        return None
    else:
        return str(v)


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser(description='Binary Triplet classification with HANAMI')
# Dataset
parser.add_argument('--data-name', default='drkg_ms', help='graph name')
parser.add_argument('--data-name1', default='ms', help='graph name')
parser.add_argument('--task', default='binary', help='graph name')

parser.add_argument('--input_dir', type=str, default='./data/')
parser.add_argument('--res_dir', type=str, default='24-7-4-binary base')

parser.add_argument('--dise_feat_dir', type=str, default='./data/drkg/DRKG_MS_dise_Rev.pth')
parser.add_argument('--drug_feat_dir', type=str, default='./data/drkg/DRKG_MS_drug_Rev.pth')
parser.add_argument('--gene_feat_dir', type=str, default='./data/drkg/DRKG_MS_gene_Rev.pth')

parser.add_argument('--test-ratio', type=float, default=0.1, help='ratio of test triplets')
parser.add_argument('--val-ratio', type=float, default=0.1, help='ratio of validation triplets')

# Model and Training
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)') #seed!
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight-decay', type=float, default=0)
parser.add_argument('--hidden-channels', type=int, default=256)
parser.add_argument('--batch-size', type=int, default=5000)
parser.add_argument('--batch_num', type=int, default=10)
parser.add_argument('--epoch-num', type=int, default=150) #!!!
parser.add_argument('--tau', type=float, default=1000)
parser.add_argument('--lam', type=float, default=0.1)

parser.add_argument('--abla_edge', action="store_true", help="whether to remove edge pooling")
parser.add_argument('--abla_basic', action="store_true", help="whether to remove basic pooling")

args = parser.parse_args()
args.input_dir1 = args.input_dir + 'ms' + '/'
args.input_dir = args.input_dir + 'drkg' + '/'

args.res_dir = './results/' + 'drkg' + ' ' + args.res_dir + '/'
if args.data_name == 'drkg':
    args.drug_num = 2908
    args.dise_num = 2157
    args.gene_num = 9809
if args.data_name == 'ms':
    args.drug_num = 1272
    args.dise_num = 694
    args.gene_num = 4519
if args.data_name == 'drkg_ms':
    args.drug_num = 1636
    args.dise_num = 1551
    args.gene_num = 5291
    
if args.data_name1 == 'ms':
    args.drug_num1 = 1272
    args.dise_num1 = 694
    args.gene_num1 = 4519
if args.data_name1 == 'drkg_ms':
    args.drug_num1 = 1636
    args.dise_num1 = 1551
    args.gene_num1 = 5291
    
def npair_loss(z1, z2, margin=1.0):
    batch_size = z1.size(0)
    device = z1.device

    # Normalize embeddings
    anchor_norm = F.normalize(z1, dim=1)
    positive_norm = F.normalize(z2, dim=1)

    # Compute similarity matrix between anchor and positive batch embeddings
    similarity_matrix = torch.matmul(anchor_norm, positive_norm.T)  # shape (batch_size, batch_size)

    # The diagonal elements are similarity of positive pairs
    positive_sim = torch.diag(similarity_matrix)  # shape (batch_size,)

    # For each anchor, compute loss using other positives as negatives:
    diff = similarity_matrix - positive_sim.unsqueeze(1)  # shape (batch_size, batch_size)

    # Exclude diagonal elements from summation by setting them to large negative number
    mask = torch.eye(batch_size, device=device).bool()
    diff.masked_fill_(mask, float('-inf'))

    loss = torch.log1p(torch.exp(diff).sum(dim=1)).mean()

    # Optional margin term to stabilize optimization
    if margin > 0:
        loss += margin * positive_sim.mean()

    return loss

def train(infeat, edge_index, pos_train_edge, neg_train_edge,edge_attr=None):
    '''
    pos_train_edge: N * 3
    '''
    
    model.train()
    total_loss = 0
    adjmask = torch.ones_like(edge_index[0], dtype=torch.bool).to(device) #homo
    
    for _ in range(batch_num):
        optimizer.zero_grad()
        loss = 0
        
        
        adj_perm = torch.randperm(len(edge_index[0]))   #homo
        adjmask[adj_perm[:batch_size]] = 0
        
        h = model(infeat,edge_index)
        h_ = model(infeat,edge_index[:, adjmask])

        contras_loss = npair_loss(h, h_)
        loss = args.lam * contras_loss

        start = len(pos_train_edge)//batch_num * _
        end = len(pos_train_edge)//batch_num * (_ + 1)
        edge = pos_train_edge[start:end]
        feat = model.pred(h, edge) #node pooling
        feat_ = model.pred(h_, edge)
        if not args.abla_edge:
            input_feat = model.pooling2(h, edge) #edge pooling
            input_feat_ = model.pooling2(h_, edge)
            
            # concat node pooling and edge pooling 
            feat_2channle = torch.cat((feat, input_feat), dim=1)
            feat_2channle_ = torch.cat((feat_, input_feat_), dim=1)
        outs = []
        outs_ = []
        if not args.abla_edge:
            outs.append(model.classifier(feat_2channle))
            outs_.append(model.classifier(feat_2channle_))
        else:
            outs.append(model.classifier(feat))
            outs_.append(model.classifier(feat_))

        lab = []
        lab.append(torch.ones(outs[-1].shape[0]).long().to(device))

        edge = neg_train_edge[start:end]

        feat = model.pred(h, edge)
        feat_ = model.pred(h_, edge)
        if not args.abla_edge:
            input_feat = model.pooling2(h, edge)
            input_feat_ = model.pooling2(h_, edge)
            
            # concat node pooling and edge pooling 
            feat_2channle = torch.cat((feat, input_feat), dim=1)
            feat_2channle_ = torch.cat((feat_, input_feat_), dim=1)

            outs.append(model.classifier(feat_2channle))
            outs_.append(model.classifier(feat_2channle_))
        else:
            outs.append(model.classifier(feat))
            outs_.append(model.classifier(feat_))

        lab.append(torch.zeros(outs[-1].shape[0]).long().to(device))
        outs = torch.cat(outs, dim=0)
        outs_ = torch.cat(outs_, dim=0)
        lab = torch.cat(lab, dim=0)
        loss += crsoftmax(outs, lab)
        if args.lam > 0:
            loss += crsoftmax(outs_, lab)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() / 7
    return total_loss / batch_num


def ttest(infeat, edge_index, pos_valid_edge, neg_valid_edge, edge_attr=None, data_type='test'):
    model.eval()
    
    with torch.no_grad():
        h = model(infeat, edge_index)
        
        edge = pos_valid_edge
        feat = model.pred(h, edge)  # pooling 1
        if not args.abla_edge:
            input_feat = model.pooling2(h, edge)  # pooling 2
            feat = torch.cat((feat, input_feat), dim=1)

        out = model.classifier(feat)
        out = torch.softmax(out, dim=1)
        prb = [out[:, 1].cpu().detach()]
        prd = [out.argmax(dim=1).cpu().detach()]

        lab = [np.ones(out.shape[0])]

        edge = neg_valid_edge
        feat = model.pred(h, edge)  # pooling 1
        if not args.abla_edge:
            input_feat = model.pooling2(h, edge)  # pooling 2
            feat = torch.cat((feat, input_feat), dim=1)

        out = model.classifier(feat)
        out = torch.softmax(out, dim=1)
        prb.append(out[:, 1].cpu().detach())
        lab.append(np.zeros(out.shape[0]))
        prd.append(out.argmax(dim=1).cpu().detach())

        prb = torch.cat(prb, dim=0).numpy()
        prd = torch.cat(prd, dim=0).numpy()
        lab = np.concatenate(lab, axis=0)
        pre_data = precision_score(lab, prd, zero_division=0.0)
        rec_data = recall_score(lab, prd)
        acc_data = accuracy_score(lab, prd)
        auc_data = roc_auc_score(lab, prb)
        precision, recall, _ = precision_recall_curve(lab, prb)
        apr_data = auc(recall, precision)
        f1_data = f1_score(lab, prd)

    res = [[pre_data, rec_data, acc_data, auc_data, apr_data,f1_data]]
    return res


def write_results(args, res):
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)
        write_form = 'w'
    else:
        write_form = 'a'
    with open(args.res_dir + 'DRKG_MS_AUC-data.txt', write_form) as f:
        for i in res['AUC'][:-1]:
            f.write(f'{i:.2f},')
        f.write(f'{res["AUC"][-1]:.2f}\n')
        f.close()
    with open(args.res_dir + 'DRKG_MS_AUPR-data.txt', write_form) as f:
        for i in res['AUPR'][:-1]:
            f.write(f'{i:.2f},')
        f.write(f'{res["AUPR"][-1]:.2f}\n')
        f.close()
    with open(args.res_dir + 'DRKG_MS_PRE-data.txt', write_form) as f:
        for i in res['precision'][:-1]:
            f.write(f'{i:.2f},')
        f.write(f'{res["precision"][-1]:.2f}\n')
        f.close()
        
    with open(args.res_dir + 'DRKG_MS_REC-data.txt', write_form) as f:
        for i in res['recall'][:-1]:
            f.write(f'{i:.2f},')
        f.write(f'{res["recall"][-1]:.2f}\n')
        f.close()
        
    with open(args.res_dir + 'DRKG_MS_ACC-data.txt', write_form) as f:
        for i in res['ACC'][:-1]:
            f.write(f'{i:.2f},')
        f.write(f'{res["ACC"][-1]:.2f}\n')
        f.close()
    with open(args.res_dir + 'DRKG_MS_F1-data.txt', write_form) as f:
        for i in res['F1'][:-1]:
            f.write(f'{i:.2f},')
        f.write(f'{res["F1"][-1]:.2f}\n')
        f.close()

def reset_adj(A, row, col):
    A[row, col] = 0.0
    A[col, row] = 0.0
    return A

seeds = [1,10,20,30,40,50,60,70,80,90] 
for seed in seeds:

    args.seed = seed
    print('<<Begin generating training data>>')
    data, data1, poslist, neglist = prepare_data(args) #util prepare_data
    
    data = data.to(device)
    data1 = data1.to(device)
    print('<<Complete generating training data>>')

    lr = args.lr
    weight_decay = args.weight_decay

    torch.cuda.empty_cache()

    torch.cuda.empty_cache()


    set_random_seed(args.seed)
    
    num_features = data.x.shape[1]
    
    hidden_channels = args.hidden_channels

    total_nodes = args.dise_num + args.drug_num + args.gene_num
    total_nodes1 = args.dise_num1 + args.drug_num1 + args.gene_num1
    
    model = GCN_binary_SAGE(in_dim=num_features, h_dim=hidden_channels, out_dim=hidden_channels, number_nodes=total_nodes,
                       args=args)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) #Adam optimization algorithm
    criterion = torch.nn.BCEWithLogitsLoss() #combines the sigmoid activation function and binary cross-entropy loss
    crsoftmax = torch.nn.CrossEntropyLoss() #combines nn.LogSoftmax and nn.NLLLoss into a single class

    batch_size = args.batch_size
    batch_num = args.batch_num

    te_res_list = []
    for i, key in enumerate(poslist[0]):
        Best_Val_from_maf1 = 0
        Best_metrics = 0
        Final_Test_AUC_from_maf1 = 0
        Final_Test_AP_from_maf1 = 0
        Final_Test_epoch_from_maf1 = 0

        model = GCN_binary_SAGE(in_dim=data.x.shape[1], h_dim=hidden_channels, out_dim=hidden_channels, number_nodes=total_nodes,
                       args=args)

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        edges_in_graph = data.train_graph
        edges_in_graph1 = data1.train_graph1
        
        adj = torch.zeros(total_nodes, total_nodes).to(device)
        adj[edges_in_graph[0], edges_in_graph[1]] = 1.0
        adj = reset_adj(adj, neglist[1][key][:, 0], neglist[1][key][:, 1])
        adj = reset_adj(adj, neglist[1][key][:, 0], neglist[1][key][:, 2])
        adj = reset_adj(adj, neglist[1][key][:, 1], neglist[1][key][:, 2])
        edges_in_graph = adj.nonzero().t()
        
        adj1 = torch.zeros(total_nodes1, total_nodes1).to(device)
        adj1[edges_in_graph1[0], edges_in_graph1[1]] = 1.0
        adj1 = reset_adj(adj1, neglist[2][key][:, 0], neglist[2][key][:, 1])
        adj1 = reset_adj(adj1, neglist[2][key][:, 0], neglist[2][key][:, 2])
        adj1 = reset_adj(adj1, neglist[2][key][:, 1], neglist[2][key][:, 2])
        edges_in_graph1 = adj1.nonzero().t()

        for epoch in range(0, args.epoch_num):
            loss_epoch = train(data.x, edges_in_graph, poslist[0][key], neglist[0][key],data.train_attr) 

            va_res = ttest(data.x, edges_in_graph, poslist[1][key], neglist[1][key], data.train_attr, data_type='val') 
            va_res_df = pd.DataFrame(va_res, columns=['precision', 'recall', 'ACC', 'AUC', 'AUPR','F1'])

            if va_res_df['AUC'].item() > Best_Val_from_maf1:
                Best_Val_from_maf1 = va_res_df['AUC'].item()
                te_res = ttest(data1.x1, edges_in_graph1, poslist[2][key], neglist[2][key], data1.train_attr1, data_type='test')
                te_res_df = pd.DataFrame(te_res, columns=['precision', 'recall', 'ACC', 'AUC', 'AUPR','F1'])
                Best_metrics = te_res[0]
            
                Final_Test_epoch_from_maf1 = epoch

        te_res_list.append(Best_metrics)
        print(i)

    Best_metrics = pd.DataFrame(te_res_list, columns=['precision', 'recall', 'ACC', 'AUC', 'AUPR','F1'], index=list(poslist[0].keys()))
    write_results(args, Best_metrics * 100)

    print('ok',seed)

