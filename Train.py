from numpy.core.fromnumeric import shape
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from Model import *
import torch
import numpy as np
from SyntheticDataset import *
from torch_geometric.loader import DataLoader
from scipy import stats
import argparse
import random
import matplotlib.pyplot as plt


class TrainDataset:
    """读取和创建训练数据集"""
    """初始化函数__init__设置了训练数据集文件的路径"""
    def __init__(self):

        self.TRAIN_DATA_PATH = os.path.join(os.getcwd(), 'data', 'train', 'dataset')


    """读取训练数据集的特征、邻接矩阵和标签。"""
    def ReadTrainFile(self):

        feature = os.path.join(self.TRAIN_DATA_PATH, 'train_dataset_feature.npy')
        adj = os.path.join(self.TRAIN_DATA_PATH, 'train_dataset_grid_adj.npy')
        label = os.path.join(self.TRAIN_DATA_PATH, 'train_dataset_label.npy')
        num_graph = len(np.array(pickle_read(feature), dtype=object))

        return feature, adj, label, num_graph


    """创建数据集并返回训练集(0.9)和测试集(0.1)。"""
    def CreateDataset(self):

        feature, adj, label, num_graph = self.ReadTrainFile()
        syn_dataset = SyntheticDataset(root='./' + 'Synthetic_Dataset', grid_adj=adj, grid_feature=feature, grid_label=label)
        # train_dataset = syn_dataset[:round(num_graph * 0.9)]       # round()对浮点数进行四舍五入取整
        # test_dataset = syn_dataset[round(num_graph * 0.9):]

        # 划分数据集
        train_dataset = syn_dataset[:round(num_graph * 0.3)] + syn_dataset[round(num_graph * 0.4):]
        test_dataset = syn_dataset[round(num_graph * 0.3):round(num_graph * 0.4)]
        # train_dataset = syn_dataset[:round(num_graph * 0.15)] + syn_dataset[round(num_graph * 0.2):round(num_graph * 0.35)] + syn_dataset[round(num_graph * 0.4):round(num_graph * 0.55)] + syn_dataset[round(num_graph * 0.6):round(num_graph * 0.75)] + syn_dataset[round(num_graph * 0.8):round(num_graph * 0.95)]
        # test_dataset = syn_dataset[round(num_graph * 0.15):round(num_graph * 0.2)] + syn_dataset[round(num_graph * 0.35):round(num_graph * 0.4)] + syn_dataset[round(num_graph * 0.55):round(num_graph * 0.6)] + syn_dataset[round(num_graph * 0.75):round(num_graph * 0.8)] + syn_dataset[round(num_graph * 0.95):round(num_graph * 1)]

        return train_dataset, test_dataset


def train():
    model.train()
    train_loss = 0  # 累积训练损失
    # 将模型设置为训练模式
    for data in train_loader:                                      # Iterate in batches over the training dataset.迭代训练数据集的批次
        data = data.to(device)                                     # 将数据移动到指定的设备（如GPU）上进行计算。
        out = model(data.x, data.edge_index, data.edge_weight)                       # 使用模型对输入数据进行前向传播
        loss = criterion(out, data.y.view(-1, 1))                  # 计算输出与真实标签之间的损失
        loss.backward()                                            # Derive gradients.计算损失函数关于模型参数的梯度。
        optimizer.step()                                           # Update parameters based on gradients.根据梯度更新模型参数。
        optimizer.zero_grad()                                      # Clear gradients清除之前计算的梯度，避免梯度在迭代过程中累积。
        train_loss += loss.item()                                  # 累加每个批次的训练损失

    average_train_loss = train_loss / len(train_loader)            # 计算平均训练损失

    return average_train_loss  # 返回平均训练损失

def test(loader):
    model.eval()                                                   # 将模型设置为评估模式，这意味着不会进行参数更新和梯度计算。
    test_loss = 0
    for data in loader:                                            # Iterate in batches over the training/test dataset.使用 loader 迭代数据集的小批次数据
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_weight)
        test_loss += criterion(out, data.y.view(-1, 1)).item()


    average_test_loss = test_loss / len(loader)                    # 计算平均测试损失
    return average_test_loss                                       # 返回平均测试损失





if __name__ == '__main__':


    TrainSets = TrainDataset()
    train_dataset, test_dataset = TrainSets.CreateDataset()
    # t = 0
    # print(train_dataset[t].x.shape)
    # print(train_dataset[t].y.shape)
    # print(train_dataset[t].edge_index.shape)
    # print(train_dataset[t].edge_weight.shape)
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SNDM().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)   # , weight_decay=0.0001使用 Adam 优化器来更新模型的参数，学习率为 0.001，权重衰减为 0.0001
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)

    criterion = torch.nn.MSELoss(reduction='mean')                                    # 采用均方误差损失函数
    # criterion = torch.nn.SmoothL1Loss(reduction='mean')

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # scheduler_1 = StepLR(optimizer, step_size=50, gamma=0.3)

    epoch_num = 100

    patience = 10

    train_losses = []
    test_losses = []
    best_test_loss = float('inf')  # 初始化 best_test_loss
    early_stop_counter = 0
    for epoch in range(epoch_num):
        # 训练过程
        train_loss = train()  # 调用 train 函数进行训练并返回平均训练损失
        train_losses.append(train_loss)

        # 测试过程
        test_loss = test(test_loader)  # 使用测试数据集进行测试，并返回平均测试损失
        test_losses.append(test_loss)

        scheduler.step(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_directory = r'D:\Python_Projects\Paper2-Network-Disintegration\Checkpoints'
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            model_save_path = os.path.join(save_directory, 'best_model.pkl')
            torch.save(model.state_dict(), model_save_path)
            early_stop_counter = 0  # 重置早停计数器
        else:
            early_stop_counter += 1  # 增加早停计数器

        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Early_Stop: {early_stop_counter}')

        if early_stop_counter == patience:
            print(f'Early stopping at epoch {epoch}')
            break

    # 绘制损失变化曲线
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.show()

