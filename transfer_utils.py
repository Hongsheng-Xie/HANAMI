import torch
import numpy as np
from torch_geometric.data import Data
from transfer_create_data import random_split, random_split1
import networkx as nx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def floor(x):
    return torch.div(x, 1, rounding_mode='trunc')


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def collect_train_graph(level: str, cycles_list, train: str, tp):
    train_pos = [[], []]
    train_attr = []
    if level == 'cycles':
        train_pos[0].append(cycles_list[train][:, 0])
        train_pos[1].append(cycles_list[train][:, 1])
        train_pos[1].append(cycles_list[train][:, 2])
        train_pos[0].append(cycles_list[train][:, 0])
        train_pos[0].append(cycles_list[train][:, 1])
        train_pos[1].append(cycles_list[train][:, 2])
        train_attr.append(torch.zeros(len(cycles_list[train][:, 0]), dtype=torch.long))
        train_attr.append(torch.zeros(len(cycles_list[train][:, 0]), dtype=torch.long) + 1)
        train_attr.append(torch.zeros(len(cycles_list[train][:, 0]), dtype=torch.long) + 2)
    if level == 'tuples':
        train_pos[0].append(cycles_list[train][:, tp[1][0]])
        train_pos[1].append(cycles_list[train][:, tp[1][1]])
        train_pos[0].append(cycles_list[train][:, tp[1][2]])
        train_pos[1].append(cycles_list[train][:, tp[1][3]])
        train_attr.append(torch.zeros(len(cycles_list[train][:, 0]), dtype=torch.long) + tp[2][0])
        train_attr.append(torch.zeros(len(cycles_list[train][:, 0]), dtype=torch.long) + tp[2][1])
    if level == 'single':
        train_pos[0].append(cycles_list[train][:, tp[1][0]])
        train_pos[1].append(cycles_list[train][:, tp[1][1]])
        train_attr.append(torch.zeros(len(cycles_list[train][:, 0]), dtype=torch.long) + tp[2])
     
    train_pos[0] = np.concatenate(train_pos[0], axis=0)
    train_pos[1] = np.concatenate(train_pos[1], axis=0)
    train_pos = np.stack((train_pos[0], train_pos[1]), axis=0)
    train_attr = torch.cat(train_attr, dim=0)
    return train_pos, train_attr


def negative_sampling(pos_list, pos_idx, drug_dise_adj, gene_dise_adj, gene_drug_adj):
    neg_n = 0
    neg_cand = []
    
    if pos_idx == 0:
        for i in range(1, len(pos_list)):
            neg_n += len(pos_list[i])
            neg_cand.append(pos_list[i])
    elif pos_idx == (len(pos_list) - 1):
        for i in range(0, len(pos_list) - 1):
            neg_n += len(pos_list[i])
            neg_cand.append(pos_list[i])

    else:
        for i in range(0, pos_idx):
            neg_n += len(pos_list[i])
            neg_cand.append(pos_list[i])
        for i in range(pos_idx + 1, len(pos_list)):
            neg_n += len(pos_list[i])
            neg_cand.append(pos_list[i])
    
    if len(pos_list[pos_idx]) <= neg_n:
        
        neg_idx = np.random.choice(neg_n, len(pos_list[pos_idx]), replace=False)
        neg_sam = np.concatenate(neg_cand, axis=0)[neg_idx]
    else:
        neg_sam = np.concatenate(neg_cand, axis=0)
        supp = []
        dise_num = len(drug_dise_adj[0])
        drug_num = len(drug_dise_adj)
        gene_num = len(gene_drug_adj)
        while len(supp) < (len(pos_list[pos_idx]) - neg_n):
            x = np.random.choice(dise_num)
            y = np.random.choice(drug_num)
            z = np.random.choice(gene_num)
            if pos_idx == 0:
                if drug_dise_adj[y, x] * gene_dise_adj[z, x] * gene_drug_adj[z, y] == 0:
                    supp.append((x, y, z))
                else:
                    continue
            if pos_idx == 1:
                if drug_dise_adj[y, x] * gene_dise_adj[z, x] == 0:
                    supp.append((x, y, z))
                elif drug_dise_adj[y, x] * gene_dise_adj[z, x] * gene_drug_adj[z, y] == 1:
                    supp.append((x, y, z))
                else:
                    continue
            if pos_idx == 2:
                if drug_dise_adj[y, x] * gene_drug_adj[z, y] == 0:
                    supp.append((x, y, z))
                elif drug_dise_adj[y, x] * gene_dise_adj[z, x] * gene_drug_adj[z, y] == 1:
                    supp.append((x, y, z))
                else:
                    continue
            if pos_idx == 3:
                if gene_dise_adj[z, x] * gene_drug_adj[z, y] == 0:
                    supp.append((x, y, z))
                elif drug_dise_adj[y, x] * gene_dise_adj[z, x] * gene_drug_adj[z, y] == 1:
                    supp.append((x, y, z))
                else:
                    continue
            if pos_idx == 4:
                if drug_dise_adj[y, x] == 0:
                    supp.append((x, y, z))
                elif drug_dise_adj[y, x] + gene_dise_adj[z, x] + gene_drug_adj[z, y] >= 2:
                    supp.append((x, y, z))
                else:
                    continue
            if pos_idx == 5:
                if gene_dise_adj[z, x] == 0:
                    supp.append((x, y, z))
                elif drug_dise_adj[y, x] + gene_dise_adj[z, x] + gene_drug_adj[z, y] >= 2:
                    supp.append((x, y, z))
                else:
                    continue
            if pos_idx == 6:
                if gene_drug_adj[z, y] == 0:
                    supp.append((x, y, z))
                elif drug_dise_adj[y, x] + gene_dise_adj[z, x] + gene_drug_adj[z, y] >= 2:
                    supp.append((x, y, z))
                else:
                    continue
        supp = np.array(supp)
        supp[:, 1] += dise_num
        supp[:, 2] += dise_num + drug_num
        neg_sam = np.concatenate((neg_sam, supp), axis=0)
    assert len(neg_sam) == len(pos_list[pos_idx])
    neg_sam = torch.from_numpy(neg_sam)
    return neg_sam

def build_heterogeneous_graph(drug_dis_adj, gene_dis_adj, gene_drug_adj):
    """
    Builds a single Graph from three bipartite adjacency matrices.
    Handles index offsets so nodes don't clash.
    
    Assumed Layout in Global Index:
    [ Drugs (0..Nd) ] -> [ Diseases (Nd..Nd+Ns) ] -> [ Genes (Nd+Ns..Total) ]
    """
    # 1. Determine sizes
    n_drugs = drug_dis_adj.shape[0]     # Rows of Drug-Disease
    n_diseases = drug_dis_adj.shape[1]  # Cols of Drug-Disease
    n_genes = gene_drug_adj.shape[0]    # Rows of Gene-Drug
    
    total_nodes = n_drugs + n_diseases + n_genes
    
    # 2. Define Offsets
    # Drug indices: 0 to n_drugs-1
    offset_dis = n_drugs                # Disease starts here
    offset_gene = n_drugs + n_diseases  # Gene starts here
    
    G = nx.Graph()
    G.add_nodes_from(range(total_nodes)) # Initialize all nodes
    
    # 3. Add Edges from Matrices (Converting local idx to global idx)
    
    # A. Drug-Disease (Rows=Drug, Cols=Disease)
    rows, cols = np.where(drug_dis_adj == 1)
    for r, c in zip(rows, cols):
        G.add_edge(r, c + offset_dis)
        
    # B. Gene-Disease (Rows=Gene, Cols=Disease)
    rows, cols = np.where(gene_dis_adj == 1)
    for r, c in zip(rows, cols):
        G.add_edge(r + offset_gene, c + offset_dis)
        
    # C. Gene-Drug (Rows=Gene, Cols=Drug)
    rows, cols = np.where(gene_drug_adj == 1)
    for r, c in zip(rows, cols):
        G.add_edge(r + offset_gene, c) # Drug is at offset 0
        
    return G

def get_reduced_similarity_features(similarity_matrix, target_dim):
    """
    Reduces an (N x N) similarity matrix to (N x target_dim)
    using SVD (Singular Value Decomposition).
    
    Why this works for Transfer Learning:
    - Raw Similarity (N x N) relies on specific node identities (Columns).
    - Reduced SVD (N x K) captures 'structural roles' (eigenvectors).
    - This allows Graph A (1551) and Graph B (1200) to both produce 
      matrices of width 'target_dim'.
    """
    # Ensure matrix is float
    similarity_matrix = similarity_matrix.float()
    
    # torch.svd_lowrank is efficient for large matrices
    # It decomposes Matrix M approx = U * diag(S) * V.T
    # We use U * S as the node embeddings/features.
    u, s, v = torch.svd_lowrank(similarity_matrix, q=target_dim)
    
    # Result shape: (N, target_dim)
    reduced_features = torch.matmul(u, torch.diag(s))
    
    return reduced_features

def get_binary_dataset(args, cycles, tuples, single,cycles1,tuples1,single1):
    # cycles1 DRKG_MS
    drug_num = args.drug_num
    dise_num = args.dise_num
    gene_num = args.gene_num

    drug_num1 = args.drug_num1
    dise_num1 = args.dise_num1
    gene_num1 = args.gene_num1
    
    #print(drug_num,dise_num,gene_num) #DRKG_MS
    #print(drug_num1,dise_num1,gene_num1) #MS
    train_pos = {}
    valid_pos = {}
    test_pos = {}
    train_neg = {}
    valid_neg = {}
    test_neg = {}
    
    neg_list = {}
    neg_list1 = {}

    pos_list = [cycles, tuples[0], tuples[1], tuples[2], single[0], single[1], single[2]]
    pos_list1 = [cycles1, tuples1[0], tuples1[1], tuples1[2], single1[0], single1[1], single1[2]]
    name_list = ['clique', '2-star 0', '2-star 1', '2-star 2', 'single 0', 'single 1', 'single 2']
    
    drug_drkg_ms = np.load(args.input_dir + 'subgraph_drug.npy', allow_pickle=True).item()
    dise_drkg_ms = np.load(args.input_dir + 'subgraph_dise.npy', allow_pickle=True).item()
    gene_drkg_ms = np.load(args.input_dir + 'subgraph_gene.npy', allow_pickle=True).item()
    
    #drug_dise_ms = np.load(args.input_dir1 + 'Compound-Disease-feat-hierarchy.npy')
    #gene_dise = np.load(args.input_dir1 + 'Gene-Disease-feat-hierarchy.npy')
    #gene_drug = np.load(args.input_dir1 + 'Gene-Compound-feat-hierarchy.npy')
    
    #print(len(drug_drkg_ms.keys()),len(drug_drkg_ms.values())) #1636
    drug_dise = np.load(args.input_dir + 'Compound-Disease-feat-hierarchy.npy') #DRKG_MS
    drug_dise1 = np.load(args.input_dir1 + 'Compound-Disease-feat-hierarchy.npy') #MS

    l1 = len(drug_dise[0])
    dd1 = []
    dd2 = []
    for i in range(l1):
        drug = drug_dise[0][i]
        dise = drug_dise[1][i]
        if drug in drug_drkg_ms.keys() and dise in dise_drkg_ms.keys():
            drug_ = drug_drkg_ms[drug]
            dise_ = dise_drkg_ms[dise]
            dd1.append(drug_)
            dd2.append(dise_)
    drug_dise = np.vstack((dd1,dd2))
    
    l1 = len(drug_dise1[0])
    dd1 = []
    dd2 = []
    for i in range(l1):
        drug = drug_dise1[0][i]
        dise = drug_dise1[1][i]
        if drug in drug_drkg_ms.keys() and dise in dise_drkg_ms.keys():
            drug_ = drug_drkg_ms[drug]
            dise_ = dise_drkg_ms[dise]
            dd1.append(drug_)
            dd2.append(dise_)
    drug_dise1 = np.vstack((dd1,dd2))
    
    #indices = np.where((drug_dise[0] == 33) & (drug_dise[1] == 1632))[0]
    #print(indices)
    gene_dise = np.load(args.input_dir + 'Gene-Disease-feat-hierarchy.npy')
    gene_dise1 = np.load(args.input_dir1 + 'Gene-Disease-feat-hierarchy.npy')
    l2 = len(gene_dise[0])
    gd1 = []
    gd2 = []
    for i in range(l2):
        g = gene_dise[0][i]
        d2 = gene_dise[1][i]
        if g in gene_drkg_ms.keys() and d2 in dise_drkg_ms.keys():
            gene = gene_drkg_ms[g]
            dise = dise_drkg_ms[d2]
            gd1.append(gene)
            gd2.append(dise)
    gene_dise = np.vstack((gd1,gd2))
    
    l2 = len(gene_dise1[0])
    gd1 = []
    gd2 = []
    for i in range(l2):
        g = gene_dise1[0][i]
        d2 = gene_dise1[1][i]
        if g in gene_drkg_ms.keys() and d2 in dise_drkg_ms.keys():
            gene = gene_drkg_ms[g]
            dise = dise_drkg_ms[d2]
            gd1.append(gene)
            gd2.append(dise)
    gene_dise1 = np.vstack((gd1,gd2))
    
    gene_drug = np.load(args.input_dir + 'Gene-Compound-feat-hierarchy.npy')
    gene_drug1 = np.load(args.input_dir1 + 'Gene-Compound-feat-hierarchy.npy')
    l3 = len(gene_drug[0])
    gd1 = []
    gd2 = []
    for i in range(l3):
        g = gene_drug[0][i]
        d1 = gene_drug[1][i]
        if g in gene_drkg_ms.keys() and d1 in drug_drkg_ms.keys():
            gene = gene_drkg_ms[g]
            drug = drug_drkg_ms[d1]
            gd1.append(gene)
            gd2.append(drug)
    gene_drug = np.vstack((gd1,gd2))
    
    l3 = len(gene_drug1[0])
    gd1 = []
    gd2 = []
    for i in range(l3):
        g = gene_drug1[0][i]
        d1 = gene_drug1[1][i]
        if g in gene_drkg_ms.keys() and d1 in drug_drkg_ms.keys():
            gene = gene_drkg_ms[g]
            drug = drug_drkg_ms[d1]
            gd1.append(gene)
            gd2.append(drug)
    gene_drug1 = np.vstack((gd1,gd2))
    
    drug_dise_adj = np.zeros((drug_num, dise_num)) 
    gene_dise_adj = np.zeros((gene_num, dise_num))
    gene_drug_adj = np.zeros((gene_num, drug_num))


    drug_dise_adj[drug_dise[0], drug_dise[1]] = 1.0 #adjacent matrix
    gene_dise_adj[gene_dise[0], gene_dise[1]] = 1.0
    gene_drug_adj[gene_drug[0], gene_drug[1]] = 1.0
    G = build_heterogeneous_graph(drug_dise_adj, gene_dise_adj, gene_drug_adj)
    
    drug_dise_adj1 = np.zeros((drug_num1, dise_num1)) 
    gene_dise_adj1 = np.zeros((gene_num1, dise_num1))
    gene_drug_adj1 = np.zeros((gene_num1, drug_num1))

    drug_dise_adj1[drug_dise1[0], drug_dise1[1]] = 1.0 #adjacent matrix
    gene_dise_adj1[gene_dise1[0], gene_dise1[1]] = 1.0
    gene_drug_adj1[gene_drug1[0], gene_drug1[1]] = 1.0
    G1 = build_heterogeneous_graph(drug_dise_adj1, gene_dise_adj1, gene_drug_adj1)
    
    for i, name in zip(range(7), name_list):
        neg_list[name] = negative_sampling(pos_list, i, drug_dise_adj, gene_dise_adj, gene_drug_adj) #utils negative_sampling

    for i, name in zip(range(7), name_list):
        neg_list1[name] = negative_sampling(pos_list1, i, drug_dise_adj1, gene_dise_adj1, gene_drug_adj1) #utils negative_sampling DRKG_MS
        
    train_graph = []
    train_graph1 = []
    train_attr = []
    train_attr1 = []
    
    cycles_list = split_trvate(args, cycles, cycles.shape[0],0.2,0) #utils split_trvate
    cycles_list1 = split_trvate(args, cycles1, cycles1.shape[0],0,1) #utils split_trvate
    train_pos['clique'] = torch.from_numpy(cycles_list['train']).to(device).long()
    valid_pos['clique'] = torch.from_numpy(cycles_list['valid']).to(device).long()
    test_pos['clique'] = torch.from_numpy(cycles_list1['test']).to(device).long()

    tuples_list = {}
    tuples_list1 = {}
    for i in range(3):
        tuples_list[i] = split_trvate(args, tuples[i], tuples[i].shape[0],0.2,0)
        tuples_list1[i] = split_trvate(args, tuples1[i], tuples1[i].shape[0],0,1)
        train_pos[f'2-star {i}'] = torch.from_numpy(tuples_list[i]['train']).to(device).long()
        valid_pos[f'2-star {i}'] = torch.from_numpy(tuples_list[i]['valid']).to(device).long()
        test_pos[f'2-star {i}'] = torch.from_numpy(tuples_list1[i]['test']).to(device).long()

    single_list = {}
    single_list1 = {}
    for i in range(3):
        single_list[i] = split_trvate(args, single[i], single[i].shape[0],0.2,0)
        single_list1[i] = split_trvate(args, single1[i], single1[i].shape[0],0,1)
        train_pos[f'single {i}'] = torch.from_numpy(single_list[i]['train']).to(device).long()
        valid_pos[f'single {i}'] = torch.from_numpy(single_list[i]['valid']).to(device).long()
        test_pos[f'single {i}'] = torch.from_numpy(single_list1[i]['test']).to(device).long()

    for name in name_list:
        tmp_list = split_trvate(args, neg_list[name], neg_list[name].shape[0],0.2,0)
        tmp_list1 = split_trvate(args, neg_list1[name], neg_list1[name].shape[0],0,1)
        train_neg[name] = tmp_list['train'].to(device)
        valid_neg[name] = tmp_list['valid'].to(device) #DRKG_MS
        test_neg[name] = tmp_list1['test'].to(device)

    edges, edge_attr = collect_train_graph('cycles', cycles_list, 'train', ())
    edges1, edge_attr1 = collect_train_graph('cycles', cycles_list1, 'test', ())

    train_graph.append(edges)
    train_graph1.append(edges1)
    train_attr.append(edge_attr)
    train_attr1.append(edge_attr1)
    
    for i in [(0, (0, 1, 0, 2), (0, 1)), (1, (0, 1, 1, 2), (0, 2)), (2, (0, 2, 1, 2), (1, 2))]:
        edges, edge_attr = collect_train_graph('tuples', tuples_list[i[0]], 'train', i)
        edges1, edge_attr1 = collect_train_graph('tuples', tuples_list1[i[0]], 'test', i)
        train_graph.append(edges)
        train_graph1.append(edges1)
        train_attr.append(edge_attr)
        train_attr1.append(edge_attr1)

    for i in [(0, (0, 1), 0), (1, (0, 2), 1), (2, (1, 2), 2)]:
        edges, edge_attr = collect_train_graph('single', single_list[i[0]], 'train', i)
        edges1, edge_attr1 = collect_train_graph('single', single_list1[i[0]], 'test', i)
        train_graph.append(edges)
        train_graph1.append(edges1)
        train_attr.append(edge_attr)
        train_attr1.append(edge_attr1)

    train_graph = torch.from_numpy(np.concatenate(train_graph, axis=1)).long()
    train_graph1 = torch.from_numpy(np.concatenate(train_graph1, axis=1)).long()
    train_attr = torch.cat(train_attr, dim=0).long()
    train_attr1 = torch.cat(train_attr1, dim=0).long()
    train_graph = torch.cat((train_graph, train_graph[[1, 0]]), dim=1)
    train_graph1 = torch.cat((train_graph1, train_graph1[[1, 0]]), dim=1)
    train_attr = torch.cat((train_attr, train_attr), dim=0)
    train_attr1 = torch.cat((train_attr1, train_attr1), dim=0)
    
    #print(train_graph)
    data = Data(edge_index=train_graph, edge_attr=train_attr)
    data1 = Data(edge_index=train_graph1, edge_attr=train_attr1)
    data.num_nodes = args.dise_num + args.drug_num + args.gene_num
    data1.num_nodes1 = args.dise_num1 + args.drug_num1 + args.gene_num1
    
    dise_feat = torch.load(args.input_dir + 'DRKG_MS_dise_Rev.pth')
    drug_feat = torch.load(args.input_dir + 'DRKG_MS_drug_Rev.pth')
    gene_feat = torch.load(args.input_dir + 'DRKG_MS_gene_Rev.pth')

    dise_feat = dise_feat.float()
    drug_feat = drug_feat.float()
    gene_feat = gene_feat.float()
    # dise_feat = F.normalize(dise_feat, dim=1)
    dise_feat = get_reduced_similarity_features(dise_feat, 128)
    
    print(dise_feat.shape)
    print(drug_feat.shape)
    print(gene_feat.shape)
    
    dise_feat_dim = dise_feat.shape[1]
    drug_feat_dim = drug_feat.shape[1]
    gene_feat_dim = gene_feat.shape[1]
    
    if args.data_name == 'drkg_ms':
        dise_feat = torch.cat((dise_feat, torch.zeros(args.dise_num, drug_feat_dim - dise_feat_dim)), dim=1) #TriMo,concat_Chem_MPNN
        gene_feat = torch.cat((gene_feat, torch.zeros(args.gene_num, drug_feat_dim - gene_feat_dim)), dim=1) #TriMo,concat_Chem_MPNN
        
    data.x = torch.cat((dise_feat, drug_feat, gene_feat), dim=0) #save similarity matrices seperately
    
    data.train_graph = train_graph
    data.train_attr = train_attr
    
    dise_feat1 = torch.load(args.input_dir1 + 'dise_Bio.pth')
    drug_feat1 = torch.load(args.input_dir1 + 'drug_All.pth')
    gene_feat1 = torch.load(args.input_dir1 + 'gene_All.pth')
    dise_feat1 = dise_feat1.float()
    drug_feat1 = drug_feat1.float()
    gene_feat1 = gene_feat1.float()
    
    dise_feat1 = get_reduced_similarity_features(dise_feat1, 128)
    
    print(dise_feat1.shape)
    print(drug_feat1.shape)
    print(gene_feat1.shape)

    dise_feat_dim1 = dise_feat1.shape[1]
    drug_feat_dim1 = drug_feat1.shape[1]
    gene_feat_dim1 = gene_feat1.shape[1]
    
    if args.data_name1 == 'ms':
        dise_feat1 = torch.cat((dise_feat1, torch.zeros(args.dise_num1, drug_feat_dim1 - dise_feat_dim1)), dim=1)
        gene_feat1 = torch.cat((gene_feat1, torch.zeros(args.gene_num1, drug_feat_dim1 - gene_feat_dim1)), dim=1)
        
    data1.x1 = torch.cat((dise_feat1, drug_feat1, gene_feat1), dim=0) #save similarity matrices seperately
    
    data1.train_graph1 = train_graph1
    data1.train_attr1 = train_attr1
    
    data.g = G
    data.g1 = G1
    
    return data, data1, (train_pos, valid_pos, test_pos), (train_neg, valid_neg, test_neg)


def split_trvate(args, row, n,val,test):
    
    perm = torch.randperm(n)
    n_v = floor(val * n).int()  # number of validation positive edges
    n_t = floor(test * n).int()  # number of test positive edges
    row = row[perm]
    row_list = {}
    row_list['valid'] = row[:n_v]
    row_list['test'] = row[n_v:n_v+n_t]
    row_list['train'] = row[n_v+n_t:]
    return row_list


def split_motif_mc(args, cycles, tuples, single):
    '''
        cycles              8:1:1
        tuples              8:1:1
            cenof dise      8:1:1
            cenof drug      8:1:1
            cenof gene      8:1:1
        single              8:1:1
            drug dise       8:1:1
            gene dise       8:1:1
            gene drug       8:1:1
    '''
    num_nodes = args.dise_num + args.drug_num + args.gene_num
    train_pos = []
    valid_pos = []
    test_pos = []
    train_lab = []
    valid_lab = []
    test_lab = []
    train_graph = []
    train_attr = []
    cycles_list = split_trvate(args, cycles, cycles.shape[0])
    train_pos.append(cycles_list['train'])
    train_lab += len(cycles_list['train']) * [0]
    valid_pos.append(cycles_list['valid'])
    valid_lab += len(cycles_list['valid']) * [0]
    test_pos.append(cycles_list['test'])
    test_lab += len(cycles_list['test']) * [0]

    tuples_list = {}
    for i in range(3):
        tuples_list[i] = split_trvate(args, tuples[i], tuples[i].shape[0])
        train_pos.append(tuples_list[i]['train'])
        train_lab += len(tuples_list[i]['train']) * [i + 1]
        valid_pos.append(tuples_list[i]['valid'])
        valid_lab += len(tuples_list[i]['valid']) * [i + 1]
        test_pos.append(tuples_list[i]['test'])
        test_lab += len(tuples_list[i]['test']) * [i + 1]

    single_list = {}
    for i in range(3):
        single_list[i] = split_trvate(args, single[i], single[i].shape[0])
        train_pos.append(single_list[i]['train'])
        train_lab += len(single_list[i]['train']) * [i + 4]
        valid_pos.append(single_list[i]['valid'])
        valid_lab += len(single_list[i]['valid']) * [i + 4]
        test_pos.append(single_list[i]['test'])
        test_lab += len(single_list[i]['test']) * [i + 4]
        
    egdes, edge_attr = collect_train_graph('cycles', cycles_list, 'train', ())
    train_graph.append(egdes)
    train_attr.append(edge_attr)

    for i in [(0, (0, 1, 2, 0), (0, 1)), (1, (0, 1, 1, 2), (0, 2)), (2, (2, 0, 1, 2), (1, 2))]:
        egdes, edge_attr = collect_train_graph('tuples', tuples_list[i[0]], 'train', i)
        train_graph.append(egdes)
        train_attr.append(edge_attr)

    for i in [(0, (0, 1), 0), (1, (2, 0), 1), (2, (1, 2), 2)]:
        egdes, edge_attr = collect_train_graph('single', single_list[i[0]], 'train', i)
        train_graph.append(egdes)
        train_attr.append(edge_attr)

    train_graph = torch.from_numpy(np.concatenate(train_graph, axis=1)).long()
    # train_graph = train_graph[[1,0]]
    train_attr = torch.cat(train_attr, dim=0).long()
    di_train_graph = train_graph
    train_graph = torch.cat((train_graph, train_graph[[1, 0]]), dim=1)
    train_attr = torch.cat((train_attr, train_attr), dim=0)

    data = Data(edge_index=train_graph, edge_attr=train_attr)
    data.num_nodes = args.dise_num + args.drug_num + args.gene_num
    dise_feat = torch.load(args.input_dir + 'dise_Bio.pth')
    drug_feat = torch.load(args.input_dir + 'drug_All.pth')
    gene_feat = torch.load(args.input_dir + 'gene_All.pth')
    
    dise_feat = dise_feat.float()

    dise_feat_dim = dise_feat.shape[1]

    drug_feat_dim = drug_feat.shape[1]
    gene_feat_dim = gene_feat.shape[1]
    if args.data_name == 'drkg':
        drug_feat = torch.cat((drug_feat, torch.zeros(args.drug_num, dise_feat_dim - drug_feat_dim)), dim=1)
        gene_feat = torch.cat((gene_feat, torch.zeros(args.gene_num, dise_feat_dim - gene_feat_dim)), dim=1)
    if args.data_name == 'ms':
        dise_feat = torch.cat((dise_feat, torch.zeros(args.dise_num, drug_feat_dim - dise_feat_dim)), dim=1)
        gene_feat = torch.cat((gene_feat, torch.zeros(args.gene_num, drug_feat_dim - gene_feat_dim)), dim=1)
    data.x = torch.cat((dise_feat, drug_feat, gene_feat), dim=0)
    data.train_graph = train_graph
    data.edge_index = train_graph
    data.di_train_graph = di_train_graph
    data.train_attr = train_attr
    
    data.train_pos = torch.from_numpy(np.concatenate(train_pos, axis=0)).long()
    data.val_pos = torch.from_numpy(np.concatenate(valid_pos, axis=0)).long()
    data.test_pos = torch.from_numpy(np.concatenate(test_pos, axis=0)).long()
    data.train_lab = torch.tensor(train_lab).long()
    data.valid_lab = torch.tensor(valid_lab).long()
    data.test_lab = torch.tensor(test_lab).long()
    print(di_train_graph.shape)
    print(train_graph.shape)
    return data


def prepare_data(args):
    set_random_seed(args.seed)
    cliques, drug_tuples, dise_tuples, gene_tuples, single_drdi, single_gedi, single_gedr = random_split(args) #DRKG_MS
    tuples = [dise_tuples, drug_tuples, gene_tuples]  # (3, N, 3)
    single = [single_drdi, single_gedi, single_gedr]  # (3, N, 3)
    
    cliques1, drug_tuples1, dise_tuples1, gene_tuples1, single_drdi1, single_gedi1, single_gedr1 = random_split1(args) #MS
    tuples1 = [dise_tuples1, drug_tuples1, gene_tuples1]  # (3, N, 3)
    single1 = [single_drdi1, single_gedi1, single_gedr1]  # (3, N, 3)
    if args.task == 'binary':
        data, data1, poslist, neglist = get_binary_dataset(args, cliques, tuples, single,cliques1,tuples1,single1) #util get_binary_dataset
        return data, data1, poslist, neglist
