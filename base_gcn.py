import torch
from torch_geometric.nn.conv import GCNConv, HeteroConv, SAGEConv, GATConv
from torch_geometric.nn.aggr import LSTMAggregation
import torch.nn as nn
import torch.nn.functional as F
from Attention import Attention
import math

class GCN(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, out_dim: int, number_nodes: int,args):
        super(GCN, self).__init__()

        self.bn = nn.BatchNorm1d(in_dim) #Batch Normalization
        self.conv1 = GCNConv(in_dim, h_dim) #Formula 1
        self.conv2 = GCNConv(h_dim, out_dim) #Formula 1

        self.general_mlp = nn.Sequential(
            nn.Linear(3 * out_dim, 2 * out_dim),
            nn.ReLU(),
            nn.Linear(2 * out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim//2)
        )
        self.general_mlp2 = nn.Sequential(
            nn.Linear(3 * out_dim, 4 * out_dim),
            nn.ReLU(),
            nn.Linear(4 * out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim // 2)
        )
        self.edge_lin1 = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim)
        )

        self.classifier = nn.Linear(in_features=out_dim, out_features=7)
        self.number_nodes = number_nodes

    def forward(self, x, edge_index):
        x = self.bn(x)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return x

    def pred(self, x, idx):

        xs = [x[idx[:, 0]], x[idx[:, 1]], x[idx[:, 2]]] #Formula 2
        xs = torch.cat(xs, dim=1)

        xs = self.general_mlp(xs) #Formula 3
        return xs

    def pooling2(self, x, idx):
        #print(idx[:, 0],x[idx[:, 0]])
        xs = [F.relu(self.edge_lin1(x[idx[:, 0]] - x[idx[:, 1]])),   #Formula 4
              F.relu(self.edge_lin1(x[idx[:, 0]] - x[idx[:, 2]])),   #Formula 5
              F.relu(self.edge_lin1(x[idx[:, 1]] - x[idx[:, 2]])),]  #Formula 6
        xs = torch.cat(xs, dim=1) #Formula 7
        xs = self.general_mlp2(xs) #Formula 8
        return xs

class GCN_binary(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, out_dim: int, number_nodes: int, args):
        super(GCN_binary, self).__init__()

        self.bn = nn.BatchNorm1d(in_dim)
        #lstm_agg = LSTMAggregation(in_channels=in_dim, out_channels=h_dim)
        self.conv1 = GCNConv(in_dim, h_dim)
        self.conv2 = GCNConv(h_dim, out_dim)
        if args.abla_edge:
            basic_out = out_dim * 3
        else:
            basic_out = out_dim // 2 * 3
        if not args.abla_basic:
            self.general_mlp = nn.Sequential(
                nn.Linear(3 * out_dim, 2 * out_dim),
                nn.ReLU(),
                nn.Linear(2 * out_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, basic_out)
            )
        else:
            self.general_mlp = nn.Identity()

        if not args.abla_edge:
            
            '''
            self.img_size = math.ceil(math.sqrt(out_dim))
            self.num_pixels = self.img_size * self.img_size
            self.out_dim = out_dim

            self.conve_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
            #self.conve_bn = nn.BatchNorm2d(32)
            
            #flattened_dim = 32 * self.num_pixels
            #target_output_dim = out_dim // 2 * 3
            
            #self.conve_lin = nn.Linear(flattened_dim, target_output_dim)
            '''
            self.general_mlp2 = nn.Sequential(
                nn.Linear(3 * out_dim, 4 * out_dim),
                nn.ReLU(),
                nn.Linear(4 * out_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim // 2 * 3)
            )
            self.edge_lin1 = nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(out_dim * 2, out_dim)
            )

        self.classifier = nn.Linear(out_dim * 3, 2)
        self.number_nodes = number_nodes

    def forward(self, x, edge_index):
        #print(x,x.shape)
        #print(edge_index,edge_index.shape) [2,17359]
        #from torch_geometric.utils import sort_edge_index

        #edge_index = sort_edge_index(edge_index, num_nodes=self.number_nodes, sort_by_row=False)
        x = self.bn(x)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def pred(self, x, idx):
        xs = [x[idx[:, 0]], x[idx[:, 1]], x[idx[:, 2]]]
        
        #print(x[idx[:, 0]].shape,x[idx[:, 1]].shape,x[idx[:, 2]].shape)
        xs = torch.cat(xs, dim=1)
        xs = self.general_mlp(xs)
        return xs

    def pooling2(self, x, idx):
        '''
        h = x[idx[:, 0]]
        r = x[idx[:, 1]]
        t = x[idx[:, 2]]
        
        batch_size = h.size(0)
        
        if self.out_dim < self.num_pixels:
            padding = self.num_pixels - self.out_dim
            h = F.pad(h, (0, padding))
            r = F.pad(r, (0, padding))
            t = F.pad(t, (0, padding))
            
        #Reshape to 2D Grids: (Batch, 1, H, W)
        h_img = h.view(batch_size, 1, self.img_size, self.img_size)
        r_img = r.view(batch_size, 1, self.img_size, self.img_size)
        t_img = t.view(batch_size, 1, self.img_size, self.img_size)
        
        stack = torch.cat([h_img, r_img, t_img], dim=1)
        
        out = self.conve_conv(stack)
        out = self.conve_bn(out)
        out = F.relu(out)
        
        #Flatten and Project
        out = out.view(batch_size, -1)
        out = self.conve_lin(out)
        
        return out
        '''
        xs = [F.relu(self.edge_lin1(x[idx[:, 0]] - x[idx[:, 1]])),
              F.relu(self.edge_lin1(x[idx[:, 0]] - x[idx[:, 2]])),
              F.relu(self.edge_lin1(x[idx[:, 1]] - x[idx[:, 2]]))]
        xs = torch.cat(xs, dim=1)
        xs = self.general_mlp2(xs)
        return xs
        

class GCN_binary_SAGE(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, out_dim: int, number_nodes: int, args):
        super(GCN_binary_SAGE, self).__init__()

        self.bn = nn.BatchNorm1d(in_dim)
        #lstm_agg = LSTMAggregation(in_channels=in_dim, out_channels=h_dim)
        self.conv1 = SAGEConv(in_dim, h_dim,aggr='max')
        self.conv2 = SAGEConv(h_dim, out_dim,aggr='sum')
        if args.abla_edge:
            basic_out = out_dim * 3
        else:
            basic_out = out_dim // 2 * 3
        if not args.abla_basic:
            self.general_mlp = nn.Sequential(
                nn.Linear(3 * out_dim, 2 * out_dim),
                nn.ReLU(),
                nn.Linear(2 * out_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, basic_out)
            )
        else:
            self.general_mlp = nn.Identity()

        if not args.abla_edge:
            self.general_mlp2 = nn.Sequential(
                nn.Linear(3 * out_dim, 4 * out_dim),
                nn.ReLU(),
                nn.Linear(4 * out_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim // 2 * 3)
            )
            self.edge_lin1 = nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(out_dim * 2, out_dim)
            )

        self.classifier = nn.Linear(out_dim * 3, 2)
        self.number_nodes = number_nodes

    def forward(self, x, edge_index):
        #print(x,x.shape)
        #print(edge_index,edge_index.shape) [2,17359]
        #from torch_geometric.utils import sort_edge_index

        #edge_index = sort_edge_index(edge_index, num_nodes=self.number_nodes, sort_by_row=False)
        x = self.bn(x)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def pred(self, x, idx):
        xs = [x[idx[:, 0]], x[idx[:, 1]], x[idx[:, 2]]]
        #print(x[idx[:, 0]].shape,x[idx[:, 1]].shape,x[idx[:, 2]].shape)
        xs = torch.cat(xs, dim=1)
        xs = self.general_mlp(xs)
        return xs

    def pooling2(self, x, idx):

        xs = [F.relu(self.edge_lin1(x[idx[:, 0]] - x[idx[:, 1]])),
              F.relu(self.edge_lin1(x[idx[:, 0]] - x[idx[:, 2]])),
              F.relu(self.edge_lin1(x[idx[:, 1]] - x[idx[:, 2]])),]
        xs = torch.cat(xs, dim=1)
        xs = self.general_mlp2(xs)
        return xs
    
class GCN_binary1(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, out_dim: int, in_dim_drug:int, in_dim_dise:int,in_dim_gene:int, 
                 number_nodes: int, args,dropout=0.5):
        # in_dim = 1792
        super(GCN_binary1, self).__init__()

        self.bn = nn.BatchNorm1d(in_dim)
        self.conv1 = GCNConv(in_dim, h_dim)
        self.conv2 = GCNConv(h_dim, out_dim)
        
        self.GCN_drug1 = GCNConv(in_dim_drug, h_dim)
        self.GCN_drug2 = GCNConv(h_dim, out_dim)
        
        self.GCN_disease1 = GCNConv(in_dim_dise, h_dim)
        self.GCN_disease2 = GCNConv(h_dim, out_dim)
        
        self.GCN_gene1 = GCNConv(in_dim_gene, h_dim)
        self.GCN_gene2 = GCNConv(h_dim, out_dim)
        
        self.drug_fusion = nn.Linear(out_dim * 2, out_dim)
        self.dise_fusion = nn.Linear(out_dim * 2, out_dim)
        self.gene_fusion = nn.Linear(out_dim * 2, out_dim)
        
        self.drug_fusion2 = nn.Linear(out_dim, out_dim)
        self.dise_fusion2 = nn.Linear(out_dim, out_dim)
        self.gene_fusion2 = nn.Linear(out_dim, out_dim)
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim // 2 * 3),
            nn.ReLU(),
            nn.Linear(out_dim // 2 * 3, out_dim)
        )
        
        self.attention = Attention(in_size=out_dim, dropout_rate=0.1)
        
        if args.abla_edge:
            basic_out = out_dim * 3
        else:
            basic_out = out_dim // 2 * 3
            
        #self.fuse_linear = nn.Linear(out_dim*3, basic_out)
        
        if not args.abla_basic:
            '''
            self.general_mlp = nn.Sequential(
                nn.Linear(6 * out_dim, 4 * out_dim),
                nn.ReLU(),
                nn.Linear(4 * out_dim, 2*out_dim),
                nn.ReLU(),
                nn.Linear(2*out_dim, 2*basic_out)
            )
            
            ''' #attention
            self.general_mlp = nn.Sequential(
                nn.Linear(3 * out_dim, 2 * out_dim),
                nn.ReLU(),
                nn.Linear(2 * out_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, basic_out)
            ) 
        else:
            self.general_mlp = nn.Identity()

        if not args.abla_edge:
            '''
            self.general_mlp2 = nn.Sequential(
                nn.Linear(6 * out_dim, 8 * out_dim),
                nn.ReLU(),
                nn.Linear(8 * out_dim, out_dim*2),
                nn.ReLU(),
                nn.Linear(out_dim*2, out_dim * 3)
            )
            ''' #attention
            self.general_mlp2 = nn.Sequential(
                nn.Linear(3 * out_dim, 4 * out_dim),
                nn.ReLU(),
                nn.Linear(4 * out_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim //2 * 3)
            )
            
            '''
            self.edge_lin1 = nn.Sequential(
                nn.Linear(2*out_dim, out_dim * 4),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(out_dim * 4, 2*out_dim)
            )
            
            ''' #attention
            self.edge_lin1 = nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(out_dim * 2, out_dim)
            ) 
        
        self.classifier = nn.Linear(out_dim * 3, 2)  #attention   
        #self.classifier = nn.Linear(out_dim * 6, 2)
        
        self.dropout = dropout
        self.number_nodes = number_nodes

    def _forward_drug(self, x, edge_index):
        # The GCN forward pass for drug embeddings
        #x = self.bn(x)
        x = self.GCN_drug1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p = 0.5, training=self.training)
        x = self.GCN_drug2(x, edge_index)
        return x
    
    def _forward_disease(self, x, edge_index):
        #x = self.bn(x)
        x = self.GCN_disease1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.GCN_disease2(x, edge_index)

        return x
    
    def _forward_gene(self, x, edge_index):
        #x = self.bn(x)
        x = self.GCN_gene1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.GCN_gene2(x, edge_index)

        return x
    
    def forward(self, x, edge_index,drug_sim_def,drug_graph_def,dise_sim_def,dise_graph_def,gene_sim_def,gene_graph_def,
                drug_sim_feat,dise_sim_feat,gene_sim_feat,
                drug_graph_feat=None,dise_graph_feat=None,gene_graph_feat=None):
        #forward(self, x, edge_index):
        
        #topology
        #torch.autograd.set_detect_anomaly(True)

        x = self.bn(x)
        x = self.conv1(x, edge_index)
        
        x = x.relu()
        #print('topo',x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        
        #similarity
        # Encode pre-defined similarity embeddings using the new methods
        emb1_sim = self._forward_drug(drug_sim_def, drug_graph_def)
        emb2_sim = self._forward_disease(dise_sim_def, dise_graph_def)
        emb3_sim = self._forward_gene(gene_sim_def, gene_graph_def)

        emb1_feat, emb2_feat, emb3_feat= None, None, None
        
        
        if drug_graph_feat is not None and dise_graph_feat is not None and gene_graph_feat is not None:
            # Encode feature-based similarity embeddings
            emb1_feat = self._forward_drug(drug_sim_feat, drug_graph_feat)
            emb2_feat = self._forward_disease(dise_sim_feat, dise_graph_feat)
            emb3_feat = self._forward_gene(gene_sim_feat, gene_graph_feat)

            # Fuse the two embeddings via concatenation and a linear layer
            #fused_drug = torch.relu(self.drug_fusion(torch.cat([emb1_sim, emb1_feat], dim=1)))
            #fused_disease = torch.relu(self.dise_fusion(torch.cat([emb2_sim, emb2_feat], dim=1)))
            #fused_gene = torch.relu(self.gene_fusion(torch.cat([emb3_sim, emb3_feat], dim=1)))
            
            # use one type of similarity matrix to test the performance
            fused_drug = torch.relu(self.drug_fusion2(emb1_feat))
            fused_disease = torch.relu(self.dise_fusion2(emb2_feat))
            fused_gene = torch.relu(self.gene_fusion2(emb3_feat))
            
            # Apply dropout
            emb1 = F.dropout(fused_drug, p=self.dropout, training=self.training)
            emb2 = F.dropout(fused_disease, p=self.dropout, training=self.training)
            emb3 = F.dropout(fused_gene, p=self.dropout, training=self.training)

        else:
            # If only one type of graph, no fusion is performed
            emb1, emb2, emb3 = emb1_sim, emb2_sim, emb3_sim
        
        # Return the final fused embeddings, along with intermediate ones for analysis
        
        #drug_feats = torch.stack([drug_out, drug_sim_out], dim=1)
        #drug_feats, att_drug = self.attention(drug_feats)

        #dis_feats = torch.stack([dis_out, dis_sim_out], dim=1)
        #dis_feats, att_dis = self.attention(dis_feats)
        
        #print(x.shape, emb1.shape,emb2.shape, emb3.shape) 
        # torch.Size([6485, 256]) torch.Size([1272, 256]) torch.Size([694, 256]) torch.Size([4519, 256])
        similarity_matrix = torch.cat([emb1, emb2, emb3], dim=0)
        #similarity_matrix = torch.cat([emb1, emb2, emb3], dim=0)
        #print(similarity_matrix.shape) [6485,256]
        
        #no attention
        #feats = torch.concat([x, similarity_matrix], dim=1)
        #feats = self.fusion_mlp(feats)
        #print(feats.shape) [6485,256]

        feats = torch.stack([x, similarity_matrix], dim=1)
        feats, att_ = self.attention(feats)
        assert not torch.isnan(feats).any(), "Attention produced NaN"
        #print(feats.sum())
        #print(feats.shape,"Attention") [6485,256]
    
        #return emb1, emb2, emb1_sim, emb1_emb2feat, emb2_sim, emb2_feat
        
        return feats

    def pred(self, x, idx):
        xs = [x[idx[:, 0]], x[idx[:, 1]], x[idx[:, 2]]]
        xs = torch.cat(xs, dim=1)
        
        xs = self.general_mlp(xs)
        return xs

    def pooling2(self, x, idx):
        #print(x[idx[:, 0]].shape)
        xs = [F.relu(self.edge_lin1(x[idx[:, 0]] - x[idx[:, 1]])),
              F.relu(self.edge_lin1(x[idx[:, 0]] - x[idx[:, 2]])),
              F.relu(self.edge_lin1(x[idx[:, 1]] - x[idx[:, 2]])),]
        xs = torch.cat(xs, dim=1)
        
        xs = self.general_mlp2(xs)
        return xs

class GCN_binary_hetero(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, out_dim: int, number_nodes: int, metadata, args):
        super().__init__()
        #GCN_binary_hetero, self
        self.conv1 = HeteroConv({edge_type: SAGEConv((-1, -1), h_dim) for edge_type in metadata[1]}, aggr='sum')
        self.conv2 = HeteroConv({edge_type: SAGEConv((-1, -1), h_dim) for edge_type in metadata[1]}, aggr='sum')
        '''
        self.conv1 = HeteroConv({
            ('disease', 'interacts', 'drug'): SAGEConv((-1, -1), h_dim),
            ('drug', 'rev_interacts', 'disease'): SAGEConv((-1, -1), h_dim),

            ('disease', 'associated', 'gene'): SAGEConv((-1, -1), h_dim),
            ('gene', 'rev_associated', 'disease'): SAGEConv((-1, -1), h_dim),

            ('drug', 'treats', 'gene'): SAGEConv((-1, -1), h_dim),
            ('gene', 'rev_treats', 'drug'): SAGEConv((-1, -1), h_dim)
        }, aggr='sum')

        # Second heterogeneous conv layer
        self.conv2 = HeteroConv({
            ('disease', 'interacts', 'drug'): SAGEConv((-1, -1), h_dim),
            ('drug', 'rev_interacts', 'disease'): SAGEConv((-1, -1), h_dim),
            ('disease', 'associated', 'gene'): SAGEConv((-1, -1), h_dim),
            ('gene', 'rev_associated', 'disease'): SAGEConv((-1, -1),h_dim),
            ('drug', 'treats', 'gene'): SAGEConv((-1, -1), h_dim),
            ('gene', 'rev_treats', 'drug'): SAGEConv((-1, -1), h_dim)
        }, aggr='sum')
'''
        # Final linear layers to project embeddings to desired output size per node type
        self.lin_dict = nn.ModuleDict({
            'disease': nn.Linear(h_dim, out_dim),
            'drug': nn.Linear(h_dim, out_dim),
            'gene': nn.Linear(h_dim, out_dim),
        })
        
        self.attention = Attention(in_size=out_dim, dropout_rate=0.1)
        
        if args.abla_edge:
            basic_out = out_dim * 3
        else:
            basic_out = out_dim // 2 * 3
        if not args.abla_basic:
            self.general_mlp = nn.Sequential(
                nn.Linear(3 * out_dim, 2 * out_dim),
                nn.ReLU(),
                nn.Linear(2 * out_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, basic_out)
            )
        else:
            self.general_mlp = nn.Identity()

        if not args.abla_edge:
            self.general_mlp2 = nn.Sequential(
                nn.Linear(3 * out_dim, 4 * out_dim),
                nn.ReLU(),
                nn.Linear(4 * out_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim // 2 * 3)
            )
            self.edge_lin1 = nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(out_dim * 2, out_dim)
            )

        self.classifier = nn.Linear(out_dim * 3, 2)
        self.number_nodes = number_nodes

    def forward(self, x_dict, edge_index_dict):
        # Layer 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(x) for k, x in x_dict.items()}
        #print(x_dict['gene'].shape)
        # Layer 2
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(x) for k, x in x_dict.items()}
        #print(x_dict['gene'].shape)
        out_dict = {k: self.lin_dict[k](x) for k, x in x_dict.items()}
        feat = torch.cat([out_dict['disease'], out_dict['drug'], out_dict['gene']], dim=0)
        return out_dict, feat

    def pred(self, x, idx):
        drug = x['drug'][idx[:,1]-694]
        dise = x['disease'][idx[:,0]]
        gene = x['gene'][idx[:,2]-694-1272]
        
        xs = [dise,drug,gene]
        #print(x[idx[:, 0]].shape,x[idx[:, 1]].shape,x[idx[:, 2]].shape)
        xs = torch.cat(xs, dim=1)
        xs = self.general_mlp(xs)
        return xs

    def pooling2(self, x, idx):
        drug = x['drug'][idx[:,1]-694]
        dise = x['disease'][idx[:,0]]
        gene = x['gene'][idx[:,2]-694-1272]
        xs = [F.relu(self.edge_lin1(dise - drug)),
              F.relu(self.edge_lin1(dise - gene)),
              F.relu(self.edge_lin1(drug - gene)),]
        xs = torch.cat(xs, dim=1)
        xs = self.general_mlp2(xs)
        return xs