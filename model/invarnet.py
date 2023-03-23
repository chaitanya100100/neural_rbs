import torch
import torch.nn as nn
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix, rotation_6d_to_matrix
import torch_scatter
import torch_geometric as pyg

from model.core import MLP, get_normalization


class InvarNet(pyg.nn.MessagePassing):
    def __init__(self, cfg):

        super(InvarNet, self).__init__()

        self.cfg = cfg

        self.node_feat_dim = cfg['node_feat_dim']
        self.hidden_dim = cfg['hidden_dim']
        self.num_hiddens = cfg['num_hiddens']
        self.rel_feat_dim = cfg['rel_feat_dim']
        self.mp_rel_idx = cfg['mp_rel_idx']
        self.num_mp_blocks = cfg['num_mp_blocks']
        self.effect_normalization = cfg['effect_normalization']
        self.inp_normalization = cfg['inp_normalization']

        activation = nn.ReLU
        act_normalization = nn.Identity
        # activation = nn.LeakyReLU
        # act_normalization = my_groupnorm

        # Node encoder
        node_enc_args = dict(input_dim=self.node_feat_dim, output_dim=self.hidden_dim, hidden_dim=self.hidden_dim, activation=activation, act_normalization=act_normalization, num_hiddens=self.num_hiddens)
        self.node_encoder = MLP(**node_enc_args)

        # Relation encoder
        rel_enc_args = dict(input_dim=self.rel_feat_dim, output_dim=self.hidden_dim, hidden_dim=self.hidden_dim, activation=activation, act_normalization=act_normalization, num_hiddens=self.num_hiddens)
        self.rel_encoder = MLP(**rel_enc_args)

        # Node message passing processor
        node_processor_args = dict(input_dim=self.hidden_dim*2, output_dim=self.hidden_dim, hidden_dim=self.hidden_dim, activation=activation, act_normalization=act_normalization, num_hiddens=self.num_hiddens)
        self.node_processors = nn.ModuleList([MLP(**node_processor_args) for _ in range(len(self.mp_rel_idx))])
        # Relation message passing processor
        rel_processor_args = dict(input_dim=self.hidden_dim*3, output_dim=self.hidden_dim, hidden_dim=self.hidden_dim, activation=activation, act_normalization=act_normalization, num_hiddens=self.num_hiddens)
        self.rel_processors = nn.ModuleList([MLP(**rel_processor_args) for _ in range(len(self.mp_rel_idx))])

        # Node decoder
        node_dec_args = dict(input_dim=self.hidden_dim, output_dim=3, hidden_dim=self.hidden_dim, activation=activation, act_normalization=act_normalization, num_hiddens=self.num_hiddens)
        self.node_dec = MLP(**node_dec_args)

        if self.effect_normalization != 'none':
            norm_fn = get_normalization(self.effect_normalization)
            self.node_encoder_norm = norm_fn(self.hidden_dim)
            self.rel_encoder_norm = norm_fn(self.hidden_dim)
            self.node_processor_norm = nn.ModuleList([norm_fn(self.hidden_dim) for _ in range(len(self.mp_rel_idx))])
            self.rel_processor_norm = nn.ModuleList([norm_fn(self.hidden_dim) for _ in range(len(self.mp_rel_idx))])
        if self.inp_normalization != 'none':
            self.node_encoder_input_norm = get_normalization(self.inp_normalization)(self.node_feat_dim)
            self.rel_encoder_input_norm = get_normalization(self.inp_normalization)(self.rel_feat_dim)

        self.use_indexing=False

    def forward_old(self, node_feats, rels, rel_feats, rel_stages):
        """InvarNet forward pass without using PyG. Just for reference."""
        # node_feats: N x node_feat_dim
        # rel_feats: N x rel_feat_dim
        # rels: R x 2
        # rel_stages: R
        # instance_idx: O+1
        num_nodes = node_feats.shape[0]
        num_rels = rels.shape[0]

        if self.inp_normalization != 'none':
            node_feats = self.node_encoder_input_norm(node_feats)
            rel_feats = self.rel_encoder_input_norm(rel_feats)

        node_latents = self.node_encoder(node_feats)
        rel_latents = self.rel_encoder(rel_feats)
        if self.effect_normalization != 'none':
            node_latents = self.node_encoder_norm(node_latents)
            rel_latents = self.rel_encoder_norm(rel_latents)

        for _ in range(self.num_mp_blocks):
            for idx, mpri in enumerate(self.mp_rel_idx):
                relidx = (rel_stages == mpri)  # size R, total number of True is r
                ridx, sidx = rels[relidx, 0], rels[relidx, 1]  # size r, r

                rli_ltn = self.rel_processors[idx](torch.cat([rel_latents[relidx], node_latents[ridx], node_latents[sidx]], 1))  # r
                if self.effect_normalization != 'none':
                    rli_ltn = self.rel_processor_norm[idx](rli_ltn)
                new_rel_latents = torch_scatter.scatter(rli_ltn, torch.where(relidx)[0], dim=0, dim_size=num_rels)  # R

                agg_new_rel_latents = torch_scatter.scatter_add(rli_ltn, ridx, dim=0, dim_size=num_nodes)  # N
                new_node_latents = self.node_processors[idx](torch.cat([node_latents, agg_new_rel_latents], 1))  # N
                if self.effect_normalization != 'none':
                    new_node_latents = self.node_processor_norm[idx](new_node_latents)

                node_latents = node_latents + new_node_latents
                rel_latents = rel_latents + new_rel_latents

        out = self.node_dec(node_latents)
        return out


    def forward(self, node_feats, rels, rel_feats, rel_stages):
        """InvarNet forward pass using PyG message passing.
        Args:
            node_feats: N x node_feat_dim
            rel_feats: N x rel_feat_dim
            rels: R x 2
            rel_stages: R
            instance_idx: O+1
        Returns:
            N x 3 velocity prediction
        """
        if self.use_indexing:
            return self.forward_old(node_feats, rels, rel_feats, rel_stages)

        num_nodes = node_feats.shape[0]
        num_rels = rels.shape[0]

        if self.inp_normalization != 'none':
            node_feats = self.node_encoder_input_norm(node_feats)
            rel_feats = self.rel_encoder_input_norm(rel_feats)

        # encode node and edge features
        node_latents = self.node_encoder(node_feats)
        rel_latents = self.rel_encoder(rel_feats)
        if self.effect_normalization != 'none':
            node_latents = self.node_encoder_norm(node_latents)
            rel_latents = self.rel_encoder_norm(rel_latents)

        # For each message passing block. We have usually 1 block.
        for _ in range(self.num_mp_blocks):

            # For each message passing relation type.
            for idx, mpri in enumerate(self.mp_rel_idx):
                
                # Select relations belonging to this relation type.
                relidx = (rel_stages == mpri)  # size R, total number of True is r

                # PyG message passing propagate function.
                new_rel_latents, agg_new_rel_latents = self.propagate(
                    edge_index=rels[relidx].t()[[1,0]],  # 2 x r edge_index
                    node_latents=(node_latents, node_latents),
                    rel_latents=rel_latents[relidx], 
                    relidx=relidx, 
                    num_rels=num_rels, num_nodes=num_nodes, idx=idx)
                
                # post process aggregated messages with current node latents.
                new_node_latents = self.node_processors[idx](torch.cat([node_latents, agg_new_rel_latents], 1))  # N
                if self.effect_normalization != 'none':
                    new_node_latents = self.node_processor_norm[idx](new_node_latents)

                # update node and relation latents using skip connection.
                node_latents = node_latents + new_node_latents
                rel_latents = rel_latents + new_rel_latents

        # decode node latents to get velocity prediction.
        out = self.node_dec(node_latents)
        return out
    
    def message(self, index, node_latents_i, node_latents_j, rel_latents, relidx, num_rels, num_nodes, idx):
        # First compute edge messages using edge latents and node latents.
        rli_ltn = self.rel_processors[idx](torch.cat([rel_latents, node_latents_i, node_latents_j], 1))  # r x d
        if self.effect_normalization != 'none':
            rli_ltn = self.rel_processor_norm[idx](rli_ltn)
        return relidx, rli_ltn, num_rels, num_nodes
        

    def aggregate(self, inputs, index, dim_size=None):
        relidx, rli_ltn, num_rels, num_nodes = inputs
        # scatter rli_ltn (for current rel type) to original relation latent matrix (for all rel types).
        # This step helps aggregation on each node using index.
        new_rel_latents = torch_scatter.scatter(rli_ltn, torch.where(relidx)[0], dim=0, dim_size=num_rels)  # R
        
        # Now aggregate messages on each node.
        agg_new_rel_latents = torch_scatter.scatter_add(rli_ltn, index, dim=0, dim_size=num_nodes)  # N
        return new_rel_latents, agg_new_rel_latents