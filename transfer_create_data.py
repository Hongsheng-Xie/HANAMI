import torch as th
import numpy as np
from tqdm import tqdm


def set_random_seed(random_seed):
    import random
    """
    Set the random seed.
    :param random_seed: Seed to be set.
    """
    th.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def sample_motif(motif, motif_type, dise_num, drug_num, drug_dise_isuse, gene_dise_isuse, gene_drug_isuse):
    sample_order = np.arange(len(motif))
    
    np.random.shuffle(sample_order)
    motif_list = []

    update = {'drug_dise_isuse_update': True,
              'gene_dise_isuse_update': True,
              'gene_drug_isuse_update': True}
    if motif_type == 'tup_of_drug':
        update['gene_dise_isuse_update'] = False
    if motif_type == 'tup_of_dise':
        update['gene_drug_isuse_update'] = False
    if motif_type == 'tup_of_gene':
        update['drug_dise_isuse_update'] = False
    count = 1
    max_drug = 0
    for i in tqdm(sample_order):
        isuse = 0
        dise, drug, gene = motif[i]
        if drug > max_drug:
            max_drug = drug
        if drug_dise_isuse[drug - dise_num, dise] == 0:
            isuse += 1
        if gene_dise_isuse[gene - (dise_num + drug_num), dise] == 0:
            isuse += 1
        if gene_drug_isuse[gene - (dise_num + drug_num), drug - dise_num] == 0:
            isuse += 1
        if isuse == 3:
            motif_list.append([dise, drug, gene])
            if update['drug_dise_isuse_update']:
                drug_dise_isuse[drug - dise_num, dise] = 1
            if update['gene_dise_isuse_update']:
                gene_dise_isuse[gene - (dise_num + drug_num), dise] = 1
            if update['gene_drug_isuse_update']:
                gene_drug_isuse[gene - (dise_num + drug_num), drug - dise_num] = 1
        count += 1
    motif_list = np.stack(motif_list, axis=0)
    return motif_list, drug_dise_isuse, gene_dise_isuse, gene_drug_isuse

def random_split(args):
    drug_num = args.drug_num
    dise_num = args.dise_num
    gene_num = args.gene_num
    #print(drug_num,dise_num,gene_num) #1636 1551 5291
    
    drug_drkg_ms = np.load(args.input_dir + 'subgraph_drug.npy', allow_pickle=True).item()
    dise_drkg_ms = np.load(args.input_dir + 'subgraph_dise.npy', allow_pickle=True).item()
    gene_drkg_ms = np.load(args.input_dir + 'subgraph_gene.npy', allow_pickle=True).item()
    
    tri_mat = np.load(args.input_dir + 'cycles.npy')
    tup_cenofdrug = np.load(args.input_dir + 'tup_cenofdrug.npy')
    tup_cenofdise = np.load(args.input_dir + 'tup_cenofdise.npy')
    tup_cenofgene = np.load(args.input_dir + 'tup_cenofgene.npy')
    
    l1 = tri_mat.shape[0]
    triple = []
    dise_num_ = 2157
    drug_num_ = 2908
    for i in range(l1):
        dise, drug, gene = tri_mat[i]
        drug = drug - dise_num_
        gene = gene - (dise_num_ + drug_num_)
        if dise in dise_drkg_ms.keys() and drug in drug_drkg_ms.keys() and gene in gene_drkg_ms.keys():
            dise = dise_drkg_ms[dise]
            drug = drug_drkg_ms[drug]
            gene = gene_drkg_ms[gene]
            
            drug = drug + dise_num
            gene = gene + (dise_num + drug_num)
            triple.append([dise,drug,gene])

    l2 = tup_cenofdrug.shape[0]
    cenofdrug = []
    for i in range(l2):
        dise, drug, gene = tup_cenofdrug[i]
        drug = drug - dise_num_
        gene = gene - (dise_num_ + drug_num_)
        if dise in dise_drkg_ms.keys() and drug in drug_drkg_ms.keys() and gene in gene_drkg_ms.keys():
            dise = dise_drkg_ms[dise]
            drug = drug_drkg_ms[drug]
            gene = gene_drkg_ms[gene]
            
            drug = drug + dise_num
            gene = gene + (dise_num + drug_num)
            cenofdrug.append([dise,drug,gene])
            
    l3 = tup_cenofdise.shape[0]
    cenofdise = []
    for i in range(l3):
        dise, drug, gene = tup_cenofdise[i]
        drug = drug - dise_num_
        gene = gene - (dise_num_ + drug_num_)
        if dise in dise_drkg_ms.keys() and drug in drug_drkg_ms.keys() and gene in gene_drkg_ms.keys():
            dise = dise_drkg_ms[dise]
            drug = drug_drkg_ms[drug]
            gene = gene_drkg_ms[gene]
            
            drug = drug + dise_num
            gene = gene + (dise_num + drug_num)
            cenofdise.append([dise,drug,gene])
            
    l4 = tup_cenofgene.shape[0]
    cenofgene = []
    for i in range(l4):
        dise, drug, gene = tup_cenofgene[i]
        drug = drug - dise_num_
        gene = gene - (dise_num_ + drug_num_)
        if dise in dise_drkg_ms.keys() and drug in drug_drkg_ms.keys() and gene in gene_drkg_ms.keys():
            dise = dise_drkg_ms[dise]
            drug = drug_drkg_ms[drug]
            gene = gene_drkg_ms[gene]
            
            drug = drug + dise_num
            gene = gene + (dise_num + drug_num)
            cenofgene.append([dise,drug,gene])

    drug_dise_isuse = np.zeros((drug_num, dise_num))
    gene_dise_isuse = np.zeros((gene_num, dise_num))
    gene_drug_isuse = np.zeros((gene_num, drug_num))

    cliques, drug_dise_isuse, gene_dise_isuse, gene_drug_isuse = sample_motif(triple, 'clique',
                                dise_num, drug_num, drug_dise_isuse, gene_dise_isuse, gene_drug_isuse)
    drug_tuples, drug_dise_isuse, gene_dise_isuse, gene_drug_isuse = sample_motif(cenofdrug, 'tup_of_drug',
                                dise_num, drug_num, drug_dise_isuse, gene_dise_isuse, gene_drug_isuse)
    dise_tuples, drug_dise_isuse, gene_dise_isuse, gene_drug_isuse = sample_motif(cenofdise, 'tup_of_dise',
                                dise_num, drug_num, drug_dise_isuse, gene_dise_isuse, gene_drug_isuse)
    gene_tuples, drug_dise_isuse, gene_dise_isuse, gene_drug_isuse = sample_motif(cenofgene, 'tup_of_gene',
                                dise_num, drug_num, drug_dise_isuse, gene_dise_isuse, gene_drug_isuse)
    
    n2 = len(cliques) * 3 + len(drug_tuples) * 2 + len(dise_tuples) * 2 + len(gene_tuples) * 2
    print("triplet&star:",n2)
    print('Cliques: {}, TupDrug: {}, TupDise: {}, TupGene: {}'.format(len(cliques), len(drug_tuples),
                                                                      len(dise_tuples), len(gene_tuples)))
    
    drug_dise = np.load(args.input_dir + 'Compound-Disease-feat-hierarchy.npy')

    l1 = len(drug_dise[0])

    drug_dise1 = []
    drug_dise2 = []
    for i in range(l1):
        drug = drug_dise[0][i]
        dise = drug_dise[1][i]
        if drug in drug_drkg_ms.keys() and dise in dise_drkg_ms.keys():
            drug_ = drug_drkg_ms[drug]
            dise_ = dise_drkg_ms[dise]
            drug_dise1.append(drug_)
            drug_dise2.append(dise_)
    drug_dise = np.vstack((drug_dise1,drug_dise2))
    
    gene_dise = np.load(args.input_dir + 'Gene-Disease-feat-hierarchy.npy')
    l2 = len(gene_dise[0])
    gene_dise1 = []
    gene_dise2 = []
    for i in range(l2):
        g = gene_dise[0][i]
        d2 = gene_dise[1][i]
        if g in gene_drkg_ms.keys() and d2 in dise_drkg_ms.keys():
            gene = gene_drkg_ms[g]
            dise = dise_drkg_ms[d2]
            gene_dise1.append(gene)
            gene_dise2.append(dise)
    gene_dise = np.vstack((gene_dise1,gene_dise2))
    
    gene_drug = np.load(args.input_dir + 'Gene-Compound-feat-hierarchy.npy')
    l3 = len(gene_drug[0])
    gene_drug1 = []
    gene_drug2 = []
    for i in range(l3):
        g = gene_drug[0][i]
        d1 = gene_drug[1][i]
        if g in gene_drkg_ms.keys() and d1 in drug_drkg_ms.keys():
            gene = gene_drkg_ms[g]
            drug = drug_drkg_ms[d1]
            gene_drug1.append(gene)
            gene_drug2.append(drug)
    gene_drug = np.vstack((gene_drug1,gene_drug2))
    
    drug_dise_adj = np.zeros((drug_num, dise_num))
    gene_dise_adj = np.zeros((gene_num, dise_num))
    gene_drug_adj = np.zeros((gene_num, drug_num))

    drug_dise_adj[drug_dise[0], drug_dise[1]] = 1.0
    gene_dise_adj[gene_dise[0], gene_dise[1]] = 1.0
    gene_drug_adj[gene_drug[0], gene_drug[1]] = 1.0

    # align the code of disease, drug and gene 
    drug_dise[0] = drug_dise[0] + dise_num
    gene_dise[0] = gene_dise[0] + dise_num + drug_num
    gene_drug[0] = gene_drug[0] + dise_num + drug_num
    gene_drug[1] = gene_drug[1] + dise_num
    
    n1 = drug_dise_adj.sum() + gene_dise_adj.sum() + gene_drug_adj.sum() # all valid edges
    
    single_drug_dise = [[], []]
    for i in range(len(drug_dise[0])):
        if drug_dise_isuse[drug_dise[0][i] - dise_num, drug_dise[1][i]] == 0:
            single_drug_dise[0].append(drug_dise[0][i])
            single_drug_dise[1].append(drug_dise[1][i])
    
    single_gene_dise = [[], []]
    for i in range(len(gene_dise[0])):
        if gene_dise_isuse[gene_dise[0][i] - dise_num - drug_num, gene_dise[1][i]] == 0:
            single_gene_dise[0].append(gene_dise[0][i])
            single_gene_dise[1].append(gene_dise[1][i])
    
    single_gene_drug = [[], []]
    for i in range(len(gene_drug[0])):
        if gene_drug_isuse[gene_drug[0][i] - dise_num - drug_num, gene_drug[1][i] - dise_num] == 0:
            single_gene_drug[0].append(gene_drug[0][i])
            single_gene_drug[1].append(gene_drug[1][i])
    
    drug_dise_negs = []
    for i in tqdm(range(len(single_drug_dise[0]))):
        drug = single_drug_dise[0][i]
        dise = single_drug_dise[1][i]

        neg_isues = False
        while not neg_isues:
            cand1 = (1 - gene_dise_adj[:, dise]).nonzero()[0]
            cand2 = (1 - gene_drug_adj[:, drug - dise_num]).nonzero()[0]
            cands_insert = np.intersect1d(cand1, cand2)
            gene = np.random.choice(cands_insert)
            
            # this gene has no relationship with this disease and drug, the [disease,drug,gene] is a single edge motif
            if gene_dise_adj[gene, dise] == 0 and gene_drug_adj[gene, drug - dise_num] == 0:
                drug_dise_negs.append(gene + dise_num + drug_num)
                neg_isues = True

    single_drdi = np.stack((single_drug_dise[1], single_drug_dise[0], drug_dise_negs), axis=1)
    print('single_drug_dise: ', single_drdi.shape)
    n2 += len(single_drdi)

    gene_dise_negs = []
    for i in tqdm(range(len(single_gene_dise[0]))):
        gene = single_gene_dise[0][i]
        dise = single_gene_dise[1][i]

        neg_isuse = False
        while not neg_isuse:
            cand1 = (1 - drug_dise_adj[:, dise]).nonzero()[0]
            cand2 = (1 - gene_drug_adj[gene - dise_num - drug_num, :]).nonzero()[0]
            cands_insert = np.intersect1d(cand1, cand2)
            drug = np.random.choice(cands_insert)
            if drug_dise_adj[drug, dise] == 0 and gene_drug_adj[gene - dise_num - drug_num, drug] == 0:
                gene_dise_negs.append(drug + dise_num)
                neg_isuse = True

    single_gedi = np.stack((single_gene_dise[1], gene_dise_negs, single_gene_dise[0]), axis=1)
    print('single_gene_dise: ', single_gedi.shape)
    n2 += len(single_gedi)

    gene_drug_negs = []
    for i in tqdm(range(len(single_gene_drug[0]))):
        gene = single_gene_drug[0][i]
        drug = single_gene_drug[1][i]

        neg_isuse = False
        while not neg_isuse:
            cand1 = (1 - drug_dise_adj[drug - dise_num, :]).nonzero()[0]
            cand2 = (1 - gene_dise_adj[gene - dise_num - drug_num, :]).nonzero()[0]
            cands_insert = np.intersect1d(cand1, cand2)
            dise = np.random.choice(cands_insert)
            if drug_dise_adj[drug - dise_num, dise] == 0 and gene_dise_adj[gene - dise_num - drug_num, dise] == 0:
                gene_drug_negs.append(dise)
                neg_isuse = True

    single_gedr = np.stack((gene_drug_negs, single_gene_drug[1], single_gene_drug[0]), axis=1)
    print('single_gene_drug: ', single_gedr.shape)
    n2 += len(single_gedr)
    assert n1 == n2
    return cliques, drug_tuples, dise_tuples, gene_tuples, single_drdi, single_gedi, single_gedr

def sample_motif1(motif, motif_type, dise_num, drug_num, drug_dise_isuse, gene_dise_isuse, gene_drug_isuse):
    sample_order = np.arange(len(motif))
    
    np.random.shuffle(sample_order)
    motif_list = []

    update = {'drug_dise_isuse_update': True,
              'gene_dise_isuse_update': True,
              'gene_drug_isuse_update': True}
    if motif_type == 'tup_of_drug':
        update['gene_dise_isuse_update'] = False
    if motif_type == 'tup_of_dise':
        update['gene_drug_isuse_update'] = False
    if motif_type == 'tup_of_gene':
        update['drug_dise_isuse_update'] = False
    count = 1
    for i in tqdm(sample_order):
        isuse = 0
        dise, drug, gene = motif[i]
        if drug_dise_isuse[drug - dise_num, dise] == 0:
            isuse += 1
        if gene_dise_isuse[gene - (dise_num + drug_num), dise] == 0:
            isuse += 1
        if gene_drug_isuse[gene - (dise_num + drug_num), drug - dise_num] == 0:
            isuse += 1
        if isuse == 3:
            motif_list.append([dise, drug, gene])
            if update['drug_dise_isuse_update']:
                drug_dise_isuse[drug - dise_num, dise] = 1
            if update['gene_dise_isuse_update']:
                gene_dise_isuse[gene - (dise_num + drug_num), dise] = 1
            if update['gene_drug_isuse_update']:
                gene_drug_isuse[gene - (dise_num + drug_num), drug - dise_num] = 1
        count += 1
    motif_list = np.stack(motif_list, axis=0)
    return motif_list, drug_dise_isuse, gene_dise_isuse, gene_drug_isuse


def random_split1(args):
    drug_num = args.drug_num1
    dise_num = args.dise_num1
    gene_num = args.gene_num1
    tri_mat = np.load(args.input_dir1 + 'cycles.npy')
    tup_cenofdrug = np.load(args.input_dir1 + 'tup_cenofdrug.npy')
    tup_cenofdise = np.load(args.input_dir1 + 'tup_cenofdise.npy')
    tup_cenofgene = np.load(args.input_dir1 + 'tup_cenofgene.npy')
    drug_dise_isuse = np.zeros((drug_num, dise_num))
    gene_dise_isuse = np.zeros((gene_num, dise_num))
    gene_drug_isuse = np.zeros((gene_num, drug_num))

    cliques, drug_dise_isuse, gene_dise_isuse, gene_drug_isuse = sample_motif1(tri_mat, 'clique',
                                dise_num, drug_num, drug_dise_isuse, gene_dise_isuse, gene_drug_isuse)
    drug_tuples, drug_dise_isuse, gene_dise_isuse, gene_drug_isuse = sample_motif1(tup_cenofdrug, 'tup_of_drug',
                                dise_num, drug_num, drug_dise_isuse, gene_dise_isuse, gene_drug_isuse)
    dise_tuples, drug_dise_isuse, gene_dise_isuse, gene_drug_isuse = sample_motif1(tup_cenofdise, 'tup_of_dise',
                                dise_num, drug_num, drug_dise_isuse, gene_dise_isuse, gene_drug_isuse)
    gene_tuples, drug_dise_isuse, gene_dise_isuse, gene_drug_isuse = sample_motif1(tup_cenofgene, 'tup_of_gene',
                                dise_num, drug_num, drug_dise_isuse, gene_dise_isuse, gene_drug_isuse)
    n2 = len(cliques) * 3 + len(drug_tuples) * 2 + len(dise_tuples) * 2 + len(gene_tuples) * 2
    print('Cliques: {}, TupDrug: {}, TupDise: {}, TupGene: {}'.format(len(cliques), len(drug_tuples),
                                                                      len(dise_tuples), len(gene_tuples)))
    
    drug_dise = np.load(args.input_dir1 + 'Compound-Disease-feat-hierarchy.npy')
    gene_dise = np.load(args.input_dir1 + 'Gene-Disease-feat-hierarchy.npy')
    gene_drug = np.load(args.input_dir1 + 'Gene-Compound-feat-hierarchy.npy')

    drug_dise_adj = np.zeros((drug_num, dise_num))
    gene_dise_adj = np.zeros((gene_num, dise_num))
    gene_drug_adj = np.zeros((gene_num, drug_num))

    drug_dise_adj[drug_dise[0], drug_dise[1]] = 1.0
    gene_dise_adj[gene_dise[0], gene_dise[1]] = 1.0
    gene_drug_adj[gene_drug[0], gene_drug[1]] = 1.0

    # align the code of disease, drug and gene 
    drug_dise[0] = drug_dise[0] + dise_num
    gene_dise[0] = gene_dise[0] + dise_num + drug_num
    gene_drug[0] = gene_drug[0] + dise_num + drug_num
    gene_drug[1] = gene_drug[1] + dise_num
    
    n1 = drug_dise_adj.sum() + gene_dise_adj.sum() + gene_drug_adj.sum() # all valid edges


    single_drug_dise = [[], []]
    for i in range(len(drug_dise[0])):
        if drug_dise_isuse[drug_dise[0][i] - dise_num, drug_dise[1][i]] == 0:
            single_drug_dise[0].append(drug_dise[0][i])
            single_drug_dise[1].append(drug_dise[1][i])

    single_gene_dise = [[], []]
    for i in range(len(gene_dise[0])):
        if gene_dise_isuse[gene_dise[0][i] - dise_num - drug_num, gene_dise[1][i]] == 0:
            single_gene_dise[0].append(gene_dise[0][i])
            single_gene_dise[1].append(gene_dise[1][i])

    single_gene_drug = [[], []]
    for i in range(len(gene_drug[0])):
        if gene_drug_isuse[gene_drug[0][i] - dise_num - drug_num, gene_drug[1][i] - dise_num] == 0:
            single_gene_drug[0].append(gene_drug[0][i])
            single_gene_drug[1].append(gene_drug[1][i])

    drug_dise_negs = []
    for i in tqdm(range(len(single_drug_dise[0]))):
        drug = single_drug_dise[0][i]
        dise = single_drug_dise[1][i]

        neg_isues = False
        while not neg_isues:
            # gene = np.random.choice(gene_num)
            cand1 = (1 - gene_dise_adj[:, dise]).nonzero()[0]
            cand2 = (1 - gene_drug_adj[:, drug - dise_num]).nonzero()[0]
            cands_insert = np.intersect1d(cand1, cand2)
            gene = np.random.choice(cands_insert)
            
            # this gene has no relationship with this disease and drug, the [disease,drug,gene] is a single edge motif
            if gene_dise_adj[gene, dise] == 0 and gene_drug_adj[gene, drug - dise_num] == 0:
                drug_dise_negs.append(gene + dise_num + drug_num)
                neg_isues = True

    single_drdi = np.stack((single_drug_dise[1], single_drug_dise[0], drug_dise_negs), axis=1)
    print('single_drug_dise_ms: ', single_drdi.shape)
    n2 += len(single_drdi)

    gene_dise_negs = []
    for i in tqdm(range(len(single_gene_dise[0]))):
        gene = single_gene_dise[0][i]
        dise = single_gene_dise[1][i]

        neg_isuse = False
        while not neg_isuse:
            # drug = np.random.choice(drug_num)
            cand1 = (1 - drug_dise_adj[:, dise]).nonzero()[0]
            cand2 = (1 - gene_drug_adj[gene - dise_num - drug_num, :]).nonzero()[0]
            cands_insert = np.intersect1d(cand1, cand2)
            drug = np.random.choice(cands_insert)
            if drug_dise_adj[drug, dise] == 0 and gene_drug_adj[gene - dise_num - drug_num, drug] == 0:
                gene_dise_negs.append(drug + dise_num)
                neg_isuse = True

    single_gedi = np.stack((single_gene_dise[1], gene_dise_negs, single_gene_dise[0]), axis=1)
    print('single_gene_dise_ms: ', single_gedi.shape)
    n2 += len(single_gedi)

    gene_drug_negs = []
    for i in tqdm(range(len(single_gene_drug[0]))):
        gene = single_gene_drug[0][i]
        drug = single_gene_drug[1][i]

        neg_isuse = False
        while not neg_isuse:
            # dise = np.random.choice(dise_num)
            cand1 = (1 - drug_dise_adj[drug - dise_num, :]).nonzero()[0]
            cand2 = (1 - gene_dise_adj[gene - dise_num - drug_num, :]).nonzero()[0]
            cands_insert = np.intersect1d(cand1, cand2)
            dise = np.random.choice(cands_insert)
            if drug_dise_adj[drug - dise_num, dise] == 0 and gene_dise_adj[gene - dise_num - drug_num, dise] == 0:
                gene_drug_negs.append(dise)
                neg_isuse = True

    single_gedr = np.stack((gene_drug_negs, single_gene_drug[1], single_gene_drug[0]), axis=1)
    print('single_gene_drug_ms: ', single_gedr.shape)
    n2 += len(single_gedr)

    assert n1 == n2
    return cliques, drug_tuples, dise_tuples, gene_tuples, single_drdi, single_gedi, single_gedr