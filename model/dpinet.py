import torch
import torch.nn as nn
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix, rotation_6d_to_matrix
from torch_scatter import scatter_add

from model.core import MLP, my_groupnorm
from model.core import RelationEncoder, ParticleEncoder, Propagator, ParticlePredictor


class DPINet(nn.Module):
    def __init__(self, cfg):

        super(DPINet, self).__init__()

        self.cfg = cfg

        node_dim = cfg['node_dim']
        node_attr_dim = cfg['node_attr_dim']
        hidden_dim = cfg['hidden_dim']
        rel_attr_dim = cfg['rel_attr_dim']
        num_stages = cfg['num_stages']
        self.num_stages = num_stages
        self.hidden_dim = hidden_dim
        self.rot_type = cfg['rot_type']
        self.impl_type = cfg['impl_type']
        self.out_type = cfg['out_type']
        self.invariant = cfg['invariant']
        assert self.impl_type in ['propnh1_relinp3', 'propnh1_relinp2', 'propnh0_relinp3', 'propnh0_relinp2', 'propnh0relu_relinp2', 'dpinet']
        assert self.rot_type in ['6d', 'quat']
        assert self.out_type in ['pos', 'vel']
        print('impl_type: ', self.impl_type, ' rot_type: ', self.rot_type, ' out_type: ', self.out_type, ' invariant: ', self.invariant)

        # node_attr also combines with offset to center-of-mass for rigids
        node_attr_dim += node_dim

        activation = nn.ReLU
        act_normalization = nn.Identity
        # activation = nn.LeakyReLU
        # act_normalization = my_groupnorm

        if self.impl_type in ['propnh1_relinp3', 'propnh1_relinp2', 'propnh0_relinp3', 'propnh0_relinp2', 'propnh0relu_relinp2']:
            # encode node from (node, node_attr) to node_enc
            node_enc_args = dict(input_dim=node_dim+node_attr_dim, output_dim=hidden_dim, hidden_dim=hidden_dim, activation=activation, act_normalization=act_normalization, num_hiddens=2)
            self.node_encs = nn.ModuleList([MLP(**node_enc_args) for _ in range(num_stages)])

            # encode relation from (r_node, r_node_attr, s_node, s_node_attr, rel_attr) to rel_enc
            rel_enc_args = dict(input_dim=2*node_dim+2*node_attr_dim+rel_attr_dim, output_dim=hidden_dim, hidden_dim=hidden_dim, activation=activation, act_normalization=act_normalization, num_hiddens=2)
            self.rel_encs = nn.ModuleList([MLP(**rel_enc_args) for _ in range(num_stages)])

            # node_prop will be initialized with zeros before propagation
            if 'propnh0' in self.impl_type: num_hidden_prop = 0
            elif 'propnh1' in self.impl_type: num_hidden_prop = 1

            if 'relinp3' in self.impl_type: input_dim_relprop = hidden_dim*3
            elif 'relinp2' in self.impl_type: input_dim_relprop = hidden_dim*2
            
            # compute relation propogation from (r_node_prop, s_node_prop, rel_enc) to rel_prop
            rel_prop_args = dict(input_dim=hidden_dim*3, output_dim=hidden_dim, hidden_dim=hidden_dim, activation=activation, act_normalization=act_normalization, num_hiddens=num_hidden_prop)
            if 'relu' in self.impl_type:
                self.rel_props = nn.ModuleList([nn.Sequential(MLP(**rel_prop_args), nn.ReLU()) for _ in range(num_stages)])
            else:
                self.rel_props = nn.ModuleList([MLP(**rel_prop_args) for _ in range(num_stages)])

            # compute node propogation from (node_enc, node_prop, rel_prop) to node_prop  (weirdly, official implementation didn't use node_prop as input)
            node_prop_args = dict(input_dim=input_dim_relprop, output_dim=hidden_dim, hidden_dim=hidden_dim, activation=activation, act_normalization=act_normalization, num_hiddens=num_hidden_prop)
            if 'relu' in self.impl_type:
                self.node_props = nn.ModuleList([nn.Sequential(MLP(**node_prop_args), nn.ReLU()) for _ in range(num_stages)])
            else:
                self.node_props = nn.ModuleList([MLP(**node_prop_args) for _ in range(num_stages)])

            # from node_prop to (quat, trans)
            self.node_output_rigid = MLP(input_dim=hidden_dim, output_dim=3+{'6d': 6, 'quat': 4}[self.rot_type], hidden_dim=hidden_dim, activation=activation, act_normalization=act_normalization, num_hiddens=2)
        else:
            nf_effect = nf_particle = 200
            nf_relation = 300
            nf_effect = nf_particle = hidden_dim
            nf_relation = hidden_dim
            
            # encode node from (node, node_attr) to node_enc
            self.node_encs = nn.ModuleList([ParticleEncoder(input_size=node_dim+node_attr_dim, output_size=nf_effect, hidden_size=nf_particle) for _ in range(num_stages)])
            # encode relation from (r_node, r_node_attr, s_node, s_node_attr, rel_attr) to rel_enc
            self.rel_encs = nn.ModuleList([RelationEncoder(input_size=2*node_dim+2*node_attr_dim+rel_attr_dim, hidden_size=nf_relation, output_size=nf_relation) for _ in range(num_stages)])
            # node_prop will be initialized with zeros before propagation
            # compute relation propogation from (r_node_prop, s_node_prop, rel_enc) to rel_prop
            self.rel_props = nn.ModuleList([Propagator(nf_relation+2*nf_effect, output_size=nf_effect) for _ in range(num_stages)])
            # compute node propogation from (node_enc, node_prop, rel_prop) to node_prop  (weirdly, official implementation didn't use node_prop as input)
            self.node_props = nn.ModuleList([Propagator(input_size=2*nf_effect, output_size=nf_effect, residual=True) for _ in range(num_stages)])
            # from node_prop to (quat, trans)
            self.node_output_rigid = ParticlePredictor(input_size=nf_effect, output_size=3+{'6d': 6, 'quat': 4}[self.rot_type], hidden_size=nf_effect)
            self.hidden_dim = nf_effect

        self.register_buffer('posvel_mean', torch.zeros(1, 6), persistent=False)
        self.register_buffer('posvel_std', torch.tensor([1, 1, 1, 0.1, 0.1, 0.1]).float()[None], persistent=False)
        self.register_buffer('quat_offset', torch.tensor([1, 0, 0, 0]).float(), persistent=False)

    def forward(self, nodes, node_attrs, rels, rel_attrs, rel_stages, prop_steps, instance_idx, dt):
        # nodes: N x node_dim
        # node_attrs: N x node_attr_dim
        # rels: R x 2
        # rel_attrs: R x rel_attr_dim
        # rel_stages: R
        # instance_idx: O+1
        num_nodes = nodes.shape[0]
        node_dim = nodes.shape[1]
        device = nodes.device
        node_effects = torch.zeros(num_nodes, self.hidden_dim, device=device)

        nodes = (nodes - self.posvel_mean) / self.posvel_std

        # node_attr also combines with offset to center-of-mass for rigids
        offsets = torch.zeros(num_nodes, node_dim, device=device)
        for i, (st, en) in enumerate(zip(instance_idx[:-1], instance_idx[1:])):
            offsets[st:en] = nodes[st:en] - nodes[st:en].mean(dim=0, keepdim=True)
        node_attrs = torch.cat([node_attrs, offsets], dim=1)

        if self.invariant:
            center = torch.cat([nodes[:, :3], torch.zeros_like(nodes[:, 3:6])], 1)
            scale = 1. / torch.tensor([0.05, 0.05, 0.05, 1., 1., 1.], device=device)[None]
        else:
            center = torch.zeros_like(nodes)
            scale = torch.ones(1, 6, device=device)

        for stage in range(self.num_stages):
            relidx = (rel_stages == stage)
            ridx, sidx = rels[relidx, 0], rels[relidx, 1]

            # encode receiver nodes
            node_enc_r = self.node_encs[stage](torch.cat([ nodes[ridx] - center[ridx], node_attrs[ridx] ], 1))
            
            # encode relations
            if self.invariant:
                inps = [ nodes[ridx] - center[ridx], node_attrs[ridx], (nodes[sidx] - center[ridx]) * scale, node_attrs[sidx], rel_attrs[relidx] ]
            else:
                inps = [ nodes[ridx], node_attrs[ridx], nodes[sidx], node_attrs[sidx], rel_attrs[relidx] ]
            rel_enc = self.rel_encs[stage](torch.cat(inps, 1))

            for ps in range(prop_steps[stage]):
                node_effect_r = node_effects[ridx]
                node_effect_s = node_effects[sidx]
                rel_effect = self.rel_props[stage](torch.cat([node_effect_r, node_effect_s, rel_enc], 1))
                rel_effect_agg_r = scatter_add(rel_effect, ridx, dim=0)[ridx]
                if 'relinp3' in self.impl_type:
                    node_effects[ridx] = node_effect_r + self.node_props[stage](torch.cat([node_enc_r, node_effect_r, rel_effect_agg_r], 1))
                elif 'relinp2' in self.impl_type:
                    node_effects[ridx] = node_effect_r + self.node_props[stage](torch.cat([node_enc_r, rel_effect_agg_r], 1))
                elif self.impl_type == 'dpinet':
                    node_effects[ridx] = self.node_props[stage](torch.cat([node_enc_r, rel_effect_agg_r], 1), res=node_effect_r)
                else:
                    raise AttributeError

        next_vel = []
        for _, (st, en) in enumerate(zip(instance_idx[:-1], instance_idx[1:])):
            pred = self.node_output_rigid(node_effects[st:en].mean(dim=0, keepdim=True))[0]  # 9
            if self.out_type == 'vel':
                pred = pred * dt

            t, q = pred[:3]*self.posvel_std[0, :3], pred[3:]
            if self.rot_type == '6d':
                R = rotation_6d_to_matrix(q)  # 3 x 3
            elif self.rot_type == 'quat':
                R = quaternion_to_matrix(q + self.quat_offset)

            p0 = nodes[st:en, :3] * self.posvel_std[:, :3] + self.posvel_mean[:, :3]
            c = p0.mean(dim=0)
            p1 = (p0 - c[None]) @ R + t[None] + c[None]
            v = (p1 - p0) / dt
            next_vel.append(v)
        next_vel = torch.cat(next_vel, 0)

        return next_vel

