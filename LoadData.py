import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
"""
测试
"""


file_path = r"D:\Python_Projects\Paper2-Network-Disintegration\data\train\30_50_BA_graph\GridAdjFile\train_adj_99.npy"
file_path1 = r"D:\Python_Projects\Paper2-Network-Disintegration\data\train\30_50_BA_graph\OptimalSetsFile\train_optimal_sets_502.npy"
file_path2 = r"D:\Python_Projects\Paper2-Network-Disintegration\data\train\30_50_BA_graph\FrequencyScoreFile\train_label_502.npy"




# file_path2 = r"D:\Python_Projects\Paper2-Network-Disintegration\data\train\30_50_BA_graph\SquareWeightsFile\train_square_weights_0.npy"
# file_path3 = r"D:\Python_Projects\Paper2-Network-Disintegration\data\train\30_50_BA_graph\NormRankLabelFile\train_label_12.npy"
#
# file_path4 = r"D:\Python_Projects\Paper2-Network-Disintegration\data\train\30_50_BA_graph\FrequencyScoreFile\train_label_1001.npy"
# file_path5 = r"D:\Python_Projects\Paper2-Network-Disintegration\data\train\30_50_BA_graph\FrequencyScoreFile\train_label_1424.npy"
# # file_path4 = r"D:\Python_Projects\Paper2-Network-Disintegration\data\train\20_50_BA_graph\GraphCoordinateFile\0.npy"
#
#
# file_path6 = r"D:\Python_Projects\Paper2-Network-Disintegration\data\train\dataset\train_dataset_Graph_adj.npy"
# file_path7 = r"D:\Python_Projects\Paper2-Network-Disintegration\data\train\dataset\train_dataset_Grid_adj.npy"
# file_path8 = r"D:\Python_Projects\Paper2-Network-Disintegration\data\train\dataset\train_dataset_feature.npy"
# file_path9 = r"D:\Python_Projects\Paper2-Network-Disintegration\data\train\dataset\train_dataset_label1.npy"
# file_path10 = r"D:\Python_Projects\Paper2-Network-Disintegration\data\train\dataset\train_dataset_label2.npy"



# file_path11 = r"D:\Python_Projects\Paper2-Network-Disintegration\data\train\20_50_BA_graph\GridFeatureMatrixFile\train_grid_feature_matrix_0.npy"
# file_path12 = r"D:\Python_Projects\Paper2-Network-Disintegration\data\train\20_50_BA_graph\FrequencyScoreFile\train_label_0.npy"
np.set_printoptions(threshold=np.inf)


train_adj_0 = nx.read_gpickle(file_path)
print("图邻接矩阵：\n", train_adj_0)
print("图节点度：\n", np.sum(train_adj_0, axis=1))


# train_optimal_sets_0 = nx.read_gpickle(file_path1)
# print("最优瓦解集：\n", train_optimal_sets_0)
#
#
# train_label_1 = nx.read_gpickle(file_path2)
# print("真实标签1：\n", train_label_1)




#
# # train_squre_weights_0 = nx.read_gpickle(file_path2)
# # print("圆内权重字典：\n", train_squre_weights_0)
# #
# train_label_0 = nx.read_gpickle(file_path4)
# print("真实标签1：\n", train_label_0)
#
# train_label_1 = nx.read_gpickle(file_path5)
# print("真实标签1：\n", train_label_1)
# # #
# train_label_1 = nx.read_gpickle(file_path3)
# print("真实标签2：\n", train_label_1)
# #
# # #
# cor = nx.read_gpickle(file_path4)
# print("图节点位置坐标：\n", cor)

# np.set_printoptions(threshold=np.inf)
#
# train_dataset_Graphadj = nx.read_gpickle(file_path6)
# print("邻接矩阵训练集：\n", train_dataset_Graphadj[0])
# print("邻接矩阵训练集：\n", train_dataset_Graphadj[1])
# print("邻接矩阵训练集：\n", train_dataset_Graphadj[150])
# print("图邻接矩阵训练集：\n", len(train_dataset_Graphadj))
#
# train_dataset_Gridadj = nx.read_gpickle(file_path7)
# print("邻接矩阵训练集：\n", train_dataset_Gridadj[0])
# print("邻接矩阵训练集：\n", train_dataset_Gridadj[1])
# print("网格邻接矩阵训练集：\n", len(train_dataset_Gridadj))
# #
# #
# train_dataset_feature = nx.read_gpickle(file_path8)
# print("特征矩阵训练集：\n", train_dataset_feature[0])
# print("特征矩阵训练集：\n", train_dataset_feature[150])

#
# train_dataset_label1 = nx.read_gpickle(file_path9)
# print("真实标签训练集，排名分数：\n", len(train_dataset_label1))
# print("真实标签训练集，排名分数：\n", train_dataset_label1[157])
# print("真实标签训练集，排名分数：\n", train_dataset_label1[150])
#
#
# train_dataset_label2 = nx.read_gpickle(file_path10)
# print("真实标签训练集，频率分数：\n", train_dataset_label2[12])
# print("真实标签训练集，频率分数：\n", train_dataset_label2[111])


# Gridfeature = nx.read_gpickle(file_path11)
# print("网格权重特征矩阵大小：\n", Gridfeature.shape)
# print("网格权重特征矩阵：\n", Gridfeature)
#
#
# G = nx.from_numpy_matrix(train_adj_0)  # 从邻接矩阵创建图
# nx.draw(G, cor, with_labels=True)  # 使用坐标绘制图
# plt.show()  # 显示图