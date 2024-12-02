import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from src.base.model import BaseModel


def get_normalized_adj(A):
    if A[0, 0] == 0:
        A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A), diag.reshape((1, -1)))
    return A_wave


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()


class UQGNN(BaseModel):
    def __init__(self, gso, blocks, Kt, Ks, dropout, feature, horizon,device,adj_mx,diffusion_step, **args):
        super(UQGNN, self).__init__(**args)
        # print(blocks)
        self.device = device
        self.adj_mx = adj_mx
        self.A_wave = None
        self.A_q = None
        self.A_h = None
        self.get_adj()

        modules = []
        for l in range(len(blocks) - 3):
            modules.append(STFusing(Kt, Ks, self.node_num, blocks[l][-1], blocks[l + 1], gso, dropout,diffusion_step,self.A_q,self.A_h))
        self.st_blocks = nn.Sequential(*modules)
        Ko = self.seq_len - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko

        self.output = GaussianNorm(Ko, blocks[-3][-1], blocks[-2], feature, self.node_num)

        self.feature = feature
        self.idx_up = torch.triu_indices(self.feature, self.feature)
        self.idx_diag = list(range(self.feature))

        self.output2 = GaussianNorm(Ko, blocks[-3][-1], blocks[-2], (feature + 1) * feature // 2, self.node_num)
        self.min_vec = 1e-4
        self.horizon = horizon




    def get_adj(self):
        self.A_wave = get_normalized_adj(self.adj_mx)
        self.A_q = torch.from_numpy(calculate_random_walk_matrix(self.A_wave).T.astype('float32'))
        self.A_h = torch.from_numpy(calculate_random_walk_matrix(self.A_wave.T).T.astype('float32'))
        self.A_wave = torch.from_numpy(self.A_wave)
        self.A_q = self.A_q.to(device=self.device)
        self.A_h = self.A_h.to(device=self.device)
        self.A_wave = self.A_wave.to(device=self.device)

    def forward(self, x, label, i=None):  # (b, t, n, f)
        origin_x = x.permute(0, 3, 1, 2)  # b,f,t,n
        x = self.st_blocks(origin_x)

        mu = self.output(x).transpose(2, 3)
        sigma = self.output2(x).transpose(2, 3)

        mu = F.softplus(mu)
        sigma = F.softplus(sigma)

        shape_s = sigma.shape
        z = torch.zeros(shape_s[0], shape_s[1], shape_s[2], self.feature, self.feature).to(device=sigma.device)
        z[..., self.idx_up[0], self.idx_up[1]] = sigma[..., :]
        z[..., self.idx_up[1], self.idx_up[0]] = sigma[..., :]

        eigval, eigvec = torch.linalg.eigh(z)
        adjusted_eigval = torch.clamp(eigval, min=self.min_vec)
        step1 = torch.matmul(eigvec, torch.diag_embed(adjusted_eigval))
        pd_matrix = torch.matmul(step1, eigvec.transpose(-2, -1))

        return mu, pd_matrix


class MDGCN2(nn.Module):
    def __init__(self, in_channels, out_channels, d_step, A_q, A_h, activation="relu"):
        super(MDGCN2, self).__init__()
        self.orders = d_step
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

        self.A_q = A_q
        self.A_h = A_h
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1.0 / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """
        batch_size, num_node, input_size, feature = X.shape
        # X.shape 0 batch_size, 1 node, 2 timestep, 3 feature
        supports = [self.A_q, self.A_h]
        x0 = X.permute(3, 1, 2, 0)  # (num_nodes, num_times, batch_size, feature)

        x0 = torch.reshape(x0, shape=[num_node, feature * input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        for support in supports:
            # x1 = torch.mm(support, x0)
            x1 = torch.matmul(support, x0)
            x = self._concat(x, x1)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.matmul(support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        x = torch.reshape(x, shape=[self.num_matrices, feature, num_node, input_size, batch_size])
        # x = torch.reshape(x, shape=[self.num_matrices, num_node, input_size, batch_size*feature])

        # x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)

        x = x.permute(4, 2, 1, 3, 0)  # batch_size, num_nodes, feature, input_size, order

        x = torch.reshape(x, shape=[batch_size, num_node, feature, input_size * self.num_matrices])

        # x = x.permute(0,2,1,3)

        # print("bing",x.shape,self.Theta1.shape)
        # x = torch.matmul(x, self.Theta1)  # (batch_size * self._num_nodes, output_size)
        # print(x.shape)
        x = torch.matmul(x, self.Theta1)

        x += self.bias

        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "selu":
            x = F.selu(x)

        # batch, node, feature, out_channels
        x = x.permute(0, 1, 3, 2)
        # print(x.shape)
        return x

class Chebyshev(nn.Module):
    def __init__(self, c_in, c_out, Ks, gso, order, A_q, A_h):
        super(Chebyshev, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(c_out))
        self.reset_parameters()
        self.order=order
        self.A_q = A_q
        self.A_h = A_h
        self.num_matrices = 2 * self.order + 1

    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):

        x = torch.permute(x, (0, 2, 3, 1))

        supports = [self.A_q, self.A_h]
        batch_size, feature, num_node, input_size = x.shape
        x0 = torch.permute(x, (2, 3, 0, 1))  # (num_nodes, num_times, batch_size, feature)
        x0 = torch.reshape(x0, shape=[num_node, feature * input_size * batch_size])
        xd = torch.unsqueeze(x0, 0)
        for support in supports:
            x1 = torch.matmul(support, x0)
            xd = self._concat(xd, x1)
            for k in range(2, self.order + 1):
                x2 = 2 * torch.matmul(support, x1) - x0
                xd = self._concat(xd, x2)/num_node
                x1, x0 = x2, x1

        xd = torch.reshape(xd, shape=[batch_size,x.shape[1],num_node, feature*input_size*self.num_matrices//x.shape[1]])
        xd=torch.narrow(xd,3,0,x.shape[-1])

        x=torch.add(x,xd)

        if self.Ks - 1 < 0:
            raise ValueError
        elif self.Ks - 1 == 0:
            x_0 = x
            x_list = [x_0]
        elif self.Ks - 1 == 1:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
        elif self.Ks - 1 >= 2:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(torch.einsum('hi,btij->bthj', 2 * self.gso, x_list[k - 1]) - x_list[k - 2])

        x = torch.stack(x_list, dim=2)

        cheb_graph_conv = torch.einsum('btkhi,kij->bthj', x, self.weight)
        cheb_graph_conv = torch.add(cheb_graph_conv, self.bias)

        return cheb_graph_conv


class MDGCN(nn.Module):
    def __init__(self, c_in, c_out, Ks, gso, order,A_q,A_h):
        super(MDGCN, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        self.Ks = Ks
        self.gso = gso
        self.cheb_graph_conv = Chebyshev(c_out, c_out, Ks, gso, order,A_q,A_h)

    def forward(self, x):
        x_gc_in = self.align(x)
        x_gc = self.cheb_graph_conv(x_gc_in)
        x_gc = x_gc.permute(0, 3, 1, 2)
        x_gc_out = torch.add(x_gc, x_gc_in)
        return x_gc_out

class ITCN(nn.Module):
    def __init__(self, Kt, c_in, c_out, node_num):
        super(ITCN, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.node_num = node_num
        self.align = Align(c_in, c_out)
        self.sigmoid=nn.Sigmoid()
        self.causal_conv = Conv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(Kt, 1),
                                  enable_padding=False, dilation=1)

    def forward(self, x):
        x_in = self.align(x)[:, :, self.Kt - 1:, :]
        x_causal_conv = self.causal_conv(x)
        x_p = x_causal_conv[:, : self.c_out, :, :]
        x_q = x_causal_conv[:, -self.c_out:, :, :]
        x = torch.mul((x_p + x_in), self.sigmoid(x_q))
        return x


class STFusing(nn.Module):
    def __init__(self, Kt, Ks, node_num, last_block_channel, channels, gso, dropout, order,A_q,A_h):
        super(STFusing, self).__init__()
        # print(channels)
        self.tmp_conv1 = ITCN(Kt, last_block_channel, channels[0], node_num)
        self.graph_conv = MDGCN(channels[0], channels[1], Ks, gso, order,A_q,A_h)
        self.tmp_conv2 = ITCN(Kt, channels[1], channels[2], node_num)
        self.tc2_ln = nn.LayerNorm([node_num, channels[2]])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # print(x.shape)
        x = self.tmp_conv1(x)
        x = self.graph_conv(x)
        x = self.relu(x)
        x = self.tmp_conv2(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)
        return x

class GaussianNorm(nn.Module):
    def __init__(self, Ko, last_block_channel, channels, end_channel, node_num):
        super(GaussianNorm, self).__init__()
        self.tmp_conv1 = ITCN(Ko, last_block_channel, channels[0], node_num)
        self.fc1 = nn.Linear(in_features=channels[0], out_features=channels[1])
        self.fc2 = nn.Linear(in_features=channels[1], out_features=end_channel)

        self.tc1_ln = nn.LayerNorm([node_num, channels[0]])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.tc1_ln(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = x.permute(0, 1, 3, 2)
        return x

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, node_num = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, node_num]).to(x)], dim=1)
        else:
            x = x
        return x


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1,
                 bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0,
                                     dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(Conv2d, self).forward(input)
        return result



