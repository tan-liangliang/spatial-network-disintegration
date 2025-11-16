# from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn as nn
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch.nn import Linear, BatchNorm1d, Dropout
# from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv, BatchNorm, Linear,GINConv



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
        # GraphSAGE卷积层可以使用边权重
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
        
        x = self.conv5(x, edge_index)
        x = self.bn5(x)
        x = self.activation(x)
        x = self.dropout(x)

 
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)

        freq_score = self.fc3(x)  # 最后一层全连接层输出freq_score

        return freq_score

