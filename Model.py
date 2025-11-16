from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn as nn
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch.nn import Linear, BatchNorm1d, Dropout

# class SNDM(torch.nn.Module):  # spatial network disintegration model
#     def __init__(self):
#         super(SNDM, self).__init__()
#
#         self.conv1 = GATConv(4, 16, heads=2, concat=True, negative_slope=0.2, dropout=0.2)  # 64 4
#         # self.lin1 = Linear(20, 20, bias=True)  # Additive bias layer
#         # self.bn1 = torch.nn.BatchNorm1d(20)
#
#         self.conv2 = GATConv(32, 16, heads=4, concat=True, negative_slope=0.2, dropout=0.2)  # 16 8
#         # self.lin2 = Linear(160, 160, bias=True)  # Additive bias layer
#         # self.bn2 = torch.nn.BatchNorm1d(160)
#
#         self.conv3 = GATConv(64, 16, heads=8, concat=True, negative_slope=0.2, dropout=0.2)
#         # self.lin3 = Linear(128, 128, bias=True)  # Additive bias layer
#         # self.bn3 = torch.nn.BatchNorm1d(128)
#
#         self.lin4 = Linear(128, 1, bias=False)
#
#         self.activation = nn.ELU()  # LeakyReLU   ELU   ReLU  PReLU
#         self.dropout = nn.Dropout(0.2)
#
#     def forward(self, x, edge_index, edge_weight):
#
#
#         # x = self.dropout(self.activation(self.bn1(self.conv1(x, edge_index, edge_weight))))
#         # x = self.dropout(self.activation(self.bn2(self.lin2(self.conv2(x, edge_index, edge_weight)))))
#         # x = self.dropout(self.activation(self.bn3(self.lin3(self.conv3(x, edge_index, edge_weight)))))
#         #
#         x = self.dropout(self.activation(self.conv1(x, edge_index, edge_weight)))
#         x = self.dropout(self.activation(self.conv2(x, edge_index, edge_weight)))
#         x = self.dropout(self.activation(self.conv3(x, edge_index, edge_weight)))
#         # # x3 = self.dropout(self.activation(self.bn3(self.lin3(self.conv3(x2, edge_index)))))
#
#         freq_score = self.lin4(x)
#         return freq_score



import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# class SNDM(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = GCNConv(3, 32)
#         self.conv2 = GCNConv(32, 64)
#         self.conv3 = GCNConv(64, 128)
#         self.lin1 = Linear(128, 1, bias=False)
#
#     def forward(self, x, edge_index):
#         # x, edge_index = data.x, data.edge_index
#
#         x = self.conv1(x, edge_index)
#         x = F.elu(x)   # relu  leaky_relu
#         x = F.dropout(x, p=0.2, training=self.training)
#
#         x = self.conv2(x, edge_index)
#         x = F.elu(x)
#         x = F.dropout(x,  p=0.2, training=self.training)
#         #
#         x = self.conv3(x, edge_index)
#         x = F.elu(x)
#         x = F.dropout(x,  p=0.2, training=self.training)
#
#         freq_score = self.lin1(x)
#         return freq_score


# class SNDM(torch.nn.Module):
#     def __init__(self):
#         super(SNDM, self).__init__()
#
#         # 创建三个名为conv1、conv2、conv3的GCNConv图卷积层，输入特征维度、输出特征维度
#         self.conv1 = GCNConv(4, 10)
#         self.conv2 = GCNConv(10, 20)
#         self.conv3 = GCNConv(20, 30)
#         self.conv4 = GCNConv(30, 50)
#         # self.conv5 = GCNConv(20, 25)
#
#
#
#         # 创建三个名为bn1、bn2、bn3的批量归一化层，输入特征维度
#         self.bn1 = BatchNorm1d(10)
#         self.bn2 = BatchNorm1d(20)
#         self.bn3 = BatchNorm1d(30)
#         self.bn4 = BatchNorm1d(50)
#         # self.bn5 = BatchNorm1d(25)
#
#
#         # 创建一个名为dropout的dropout层，丢弃率为0.5
#         self.dropout = nn.Dropout(0.4)
#
#         # 创建多层感知机的全连接层
#         self.fc1 = nn.Linear(50, 25)  # 第一个全连接层
#         # self.fc2 = nn.Linear(64, 32)   # 第二个全连接层
#         self.fc3 = nn.Linear(25, 1)    # 第三个全连接层，输出维度为1
#
#         # 激活函数
#         self.activation = nn.LeakyReLU()
#
#     def forward(self, x, edge_index, edge_weight):
#         # 图卷积层和批量归一化层
#         x = self.conv1(x, edge_index, edge_weight)
#         x = self.bn1(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#
#         x = self.conv2(x, edge_index, edge_weight)
#         x = self.bn2(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#
#         x = self.conv3(x, edge_index, edge_weight)
#         x = self.bn3(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#
#         x = self.conv4(x, edge_index, edge_weight)
#         x = self.bn4(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#
#         # x = self.conv5(x, edge_index, edge_weight)
#         # x = self.bn5(x)
#         # x = self.activation(x)
#         # x = self.dropout(x)
#
#         # 多层感知机的全连接层
#         x = self.fc1(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#
#         # x = self.fc2(x)
#         # x = self.activation(x)
#         # x = self.dropout(x)
#
#         freq_score = self.fc3(x)  # 最后一层全连接层输出freq_score
#         return freq_score





import torch
from torch.nn import Linear
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

# class SNDM(torch.nn.Module):  # spatial network disintegration model
#     def __init__(self):
#         super(SNDM, self).__init__()
#
#         self.conv1 = GCNConv(3, 256)
#         self.conv2 = GCNConv(256, 128)
#         self.conv3 = GCNConv(128, 64)
#         # self.conv4 = GCNConv(64, 32)
#
#         self.lin1 = Linear(64, 1, bias=False)
#
#         self.activation = nn.ELU()
#         self.dropout = nn.Dropout(0.5)
#
#     def forward(self, x, edge_index, edge_weight):
#
#         x = self.activation(self.conv1(x, edge_index, edge_weight))
#         x = self.activation(self.conv2(x, edge_index, edge_weight))
#         x = self.activation(self.conv3(x, edge_index, edge_weight))
#         # x = self.activation(self.conv4(x, edge_index, edge_weight))
#         # x = F.dropout(x, p=0.2, training=self.training)
#         freq_score = self.lin1(x)
#
#         return freq_score

import torch
from torch.nn import Linear
from torch.nn import functional as F
from torch_geometric.nn import GINConv

# class SNDM(torch.nn.Module):  # spatial network disintegration model
#     def __init__(self):
#         super(SNDM, self).__init__()
#
#         nn1 = Linear(3, 64)
#         self.conv1 = GINConv(nn1)
#         nn2 = Linear(64, 128)
#         self.conv2 = GINConv(nn2)
#
#
#
#         self.lin1 = Linear(128, 1, bias=False)
#
#         self.activation = torch.nn.LeakyReLU()
#         self.dropout = torch.nn.Dropout(0.2)
#
#     def forward(self, x, edge_index, edge_weight = None):
#
#         x = self.activation(self.conv1(x, edge_index))
#         x = self.activation(self.conv2(x, edge_index))
#         x = F.dropout(x, p=0.2, training=self.training)
#         freq_score = self.lin1(x)
#
#         return freq_score


# class SNDM(torch.nn.Module):  # Deep Neural Network
#     def __init__(self):
#         super(SNDM, self).__init__()
#
#         self.lin1 = torch.nn.Linear(15, 128)
#         self.lin2 = torch.nn.Linear(128, 256)
#         self.lin3 = torch.nn.Linear(256, 1)
#
#         self.activation = torch.nn.ELU()
#         self.dropout = torch.nn.Dropout(0.5)
#
#     def forward(self, x):
#         x1 = self.dropout(self.activation(self.lin1(x)))
#         x2 = self.dropout(self.activation(self.lin2(x1)))
#         output = self.lin3(x2)
#         return output


import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm, Linear,GINConv


import torch


class SNDM(torch.nn.Module):
    def __init__(self):
        super(SNDM, self).__init__()
        # 使用SAGEConv图卷积层替代GCNConv
        self.conv1 = SAGEConv(4, 10)
        self.conv2 = SAGEConv(10, 20)
        self.conv3 = SAGEConv(20, 30)
        self.conv4 = SAGEConv(30, 40)
        self.conv5 = SAGEConv(40, 50)
        # self.conv6 = SAGEConv(50, 60)

        # 批量归一化层和dropout层保持不变
        self.bn1 = BatchNorm(10)
        self.bn2 = BatchNorm(20)
        self.bn3 = BatchNorm(30)
        self.bn4 = BatchNorm(40)
        self.bn5 = BatchNorm(50)
        # self.bn6 = BatchNorm(60)

        self.dropout = Dropout(0.1)

        # 全连接层和激活函数保持不变
        # self.lin1 = Linear(64, 1, bias=False)
        self.activation = torch.nn.ELU()  # ELU ReLU PReLU

        # 创建多层感知机的全连接层
        self.fc1 = nn.Linear(50, 30)  # 第一个全连接层
        self.fc2 = nn.Linear(30, 10)   # 第二个全连接层
        self.fc3 = nn.Linear(10, 1)    # 第三个全连接层，输出维度为1


    def forward(self, x, edge_index, edge_weight):
        # GraphSAGE卷积层现在可以使用边权重
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = self.activation(x)
        x = self.dropout(x)
        #
        x = self.conv5(x, edge_index)
        x = self.bn5(x)
        x = self.activation(x)
        x = self.dropout(x)

        # x = self.conv6(x, edge_index)
        # x = self.bn6(x)
        # x = self.activation(x)
        # x = self.dropout(x)


        # 多层感知机的全连接层
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        #
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)

        freq_score = self.fc3(x)  # 最后一层全连接层输出freq_score

        return freq_score




# import torch
# import torch.nn as nn
# from torch_geometric.nn import GATConv, BatchNorm
#
# class SNDM(torch.nn.Module):
#     def __init__(self):
#         super(SNDM, self).__init__()
#         # 使用GATConv图卷积层替代SAGEConv
#
#         self.conv1 = GATConv(4, 10, heads=1)
#         self.conv2 = GATConv(10, 20, heads=1)
#         self.conv3 = GATConv(20, 30, heads=1)
#         self.conv4 = GATConv(30, 40, heads=1)
#         self.conv5 = GATConv(40, 50, heads=1)
#
#
#         # 批量归一化层和dropout层保持不变
#         self.bn1 = BatchNorm(10)
#         self.bn2 = BatchNorm(20)
#         self.bn3 = BatchNorm(30)
#         self.bn4 = BatchNorm(40)
#         self.bn5 = BatchNorm(50)
#
#         self.dropout = Dropout(0.1)
#
#         # 全连接层和激活函数保持不变
#         self.activation = torch.nn.ELU()  # ELU ReLU PReLU
#
#         # 创建多层感知机的全连接层
#         self.fc1 = nn.Linear(50, 30)  # 第一个全连接层
#         self.fc2 = nn.Linear(30, 10)   # 第二个全连接层
#         self.fc3 = nn.Linear(10, 1)    # 第三个全连接层，输出维度为1
#
#     def forward(self, x, edge_index, edge_weight=None):
#         # GAT卷积层现在可以使用边权重
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#
#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#
#         x = self.conv3(x, edge_index)
#         x = self.bn3(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#
#         x = self.conv4(x, edge_index)
#         x = self.bn4(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#
#         x = self.conv5(x, edge_index)
#         x = self.bn5(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#
#         # x = self.conv6(x, edge_index)
#         # x = self.bn6(x)
#         # x = self.activation(x)
#         # x = self.dropout(x)
#
#         # 多层感知机的全连接层
#         x = self.fc1(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#
#         x = self.fc2(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#
#         freq_score = self.fc3(x)  # 最后一层全连接层输出freq_score
#
#         return freq_score
