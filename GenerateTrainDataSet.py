import numpy as np
import networkx as nx
import os
from ReadSave import *
import copy
from itertools import combinations
import math
import collections
from CentralityMenthod import *
import time
epsilon = 1e-9  # 定义一个小的epsilon值


def Generate_Graph(g_type, g_layout, num_min = 30, num_max = 50):
    """随机生成节点数在30-50之间连通无环的合成网络：ER，WS，BA"""

    num_nodes = np.random.randint(num_max - num_min + 1) + num_min
    while True:
        if g_type == 'erdos_renyi':
            g = nx.erdos_renyi_graph( n=num_nodes, p=0.1)
        elif g_type == 'small-world':
            # g = nx.connected_watts_strogatz_graph(n=num_nodes, k=4, p=0.1)
            g = nx.newman_watts_strogatz_graph(num_nodes, 2, 0.5)
        elif g_type == 'barabasi_albert':
            g = nx.barabasi_albert_graph(n=num_nodes, m=2)
        if nx.is_connected(g):
            break

    num_nodes = len(g.nodes)
    if g_layout == 'random_layout':
        layout = nx.random_layout(g)  # 随机布局
        node_coordinate = {node: list(round(coord, 4) for coord in layout[node]) for node in g.nodes()}
    elif g_layout == 'circular_layout':
        layout = nx.circular_layout(g)
        coords = np.array(list(layout.values()))
        normalized_coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0))
        node_coordinate = {i: coord.tolist() for i, coord in enumerate(normalized_coords)}
    elif g_layout == 'kamada_kawai_layout':
        layout = nx.kamada_kawai_layout(g)
        coords = np.array(list(layout.values()))
        normalized_coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0))
        node_coordinate = {i: coord.tolist() for i, coord in enumerate(normalized_coords)}
    elif g_layout == 'spectral_layout':
        layout = nx.spectral_layout(g)
        coords = np.array(list(layout.values()))
        normalized_coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0))
        node_coordinate = {i: coord.tolist() for i, coord in enumerate(normalized_coords)}
    elif g_layout == 'spiral_layout':
        layout = nx.spiral_layout(g)
        coords = np.array(list(layout.values()))
        normalized_coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0))
        node_coordinate = {i: coord.tolist() for i, coord in enumerate(normalized_coords)}

    return g, num_nodes, node_coordinate




def CreateGrid(interval=1/5):
    """生成网格字典,键为网格索引，值为网格四个点的坐标"""
    part = int(1 / interval)  # 生成一个 part × part 的二维网格
    grid = [[[(round(interval * m, 4), round(interval * n, 4)),
              (round(interval * (m + 1), 4), round(interval * n, 4)),
              (round(interval * m, 4), round(interval * (n + 1), 4)),
              (round(interval * (m + 1), 4), round(interval * (n + 1), 4))]
              for n in range(part)] for m in range(part)]

    # 将二维网格转换为一维列表
    Grid_flat = [coord for sublist in grid for coord in sublist]
    # 给每个网格单元添加一个索引
    Grid = {i: Grid_flat[i] for i in range(len(Grid_flat))}

    return Grid




def Get_Grid_Adj(Grid, cen_edge):
    size = len(Grid)
    Grid_adj = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            if i != j and len(set(cen_edge[i]).intersection(cen_edge[j])) > 0:
                Grid_adj[i][j] = len(set(cen_edge[i]).intersection(cen_edge[j]))
                Grid_adj[j][i] = len(set(cen_edge[i]).intersection(cen_edge[j]))
    return Grid_adj




def GenerateTrainData(id, graph_type, graph_layout):
    print('-------------------------------------------------------------------------------------------------------')
    print(f'Generating No.{id} training {graph_type} graphs {graph_layout} layout.')         

    DATASET_PATH = os.path.join(os.getcwd(), 'data', 'train', '30_50_'+graph_type+'_graph')  
    os.makedirs(DATASET_PATH, exist_ok=True)                                                  

    if graph_type == 'ER':
        g_type = 'erdos_renyi'
    elif graph_type == 'WS':
        g_type = 'small-world'
    elif graph_type == 'BA':
        g_type = 'barabasi_albert'


    if graph_layout == 'RL':
        g_layout = 'random_layout'
    elif graph_layout == 'CL':
        g_layout = 'circular_layout'
    elif graph_layout == 'KKL':
        g_layout = 'kamada_kawai_layout'
    elif graph_layout == 'SPEL':
        g_layout = 'spectral_layout'
    elif graph_layout == 'SPIL':
        g_layout = 'spiral_layout'


    g, num_nodes, node_positions = Generate_Graph(g_type = g_type, g_layout=g_layout)  # Generate Graph 生成图信息以.gml格式保存
    Save_Graph_GML(g, DATASET_PATH, id)
    Save_Graph_Coordinate(node_positions, DATASET_PATH, id)

    g_adjacent_matrix = np.array(nx.adjacency_matrix(g).todense())
    Save_Graph_Adj(g_adjacent_matrix, DATASET_PATH, id)

    Grid = CreateGrid()
    cen_edge, weight = FindEdgesInGrid(g, Grid, node_positions)
    Grid_Adj = Get_Grid_Adj(Grid, cen_edge)
    Save_Grid_Adj(Grid_Adj, DATASET_PATH, id)

    Save_SquareWeights(weight, DATASET_PATH, id)

    optimal_sets = ExhaustiveSearch(g, Grid, cen_edge, top_num=5)
    # optimal_sets = ExhaustiveSearch(g, Grid, cen_edge)

    print("Optimal Sets: ", optimal_sets)
    Save_OptimalSets(optimal_sets, DATASET_PATH, id)
    print(f'No.{id} Graph has searched its optimal solutions: {len(optimal_sets)} sets')
    # print('-------------------------------------------------------------------------------')

    score = Initial_Lable(optimal_sets)
    Save_FrequencyLabel(score, DATASET_PATH, id)

    # Label = Label_Normalied_Rank(score, weight)
    # Save_NormRankLabel(Label, DATASET_PATH, id)

    Grid_features = Generate_Grid_Feature(g, Grid, node_positions)
    Save_GridFeatureMatrix(Grid_features, DATASET_PATH, id)



def is_close(a, b, epsilon):
    return abs(a - b) < epsilon

def FindEdgesInGrid(net, Grid, pos):
    edge_list = list(net.edges)
    cen_edge, weight = {}, {}
    grid_index = list(Grid.keys())
    # print("网格索引：", grid_index)
    grid_pos = list(Grid.values())
    # print("网格四个点坐标：", grid_pos)
    for i in grid_index:
        weight[i] = 0
        cen_edge[i] = []
        for edge in edge_list:
            cor = [pos[edge[0]], pos[edge[1]]]  # 判断至少有一个端点在方格内
            if (grid_pos[i][0][0] <= cor[0][0] <= grid_pos[i][1][0] and grid_pos[i][0][1] <= cor[0][1] <= grid_pos[i][2][1]) or \
               (grid_pos[i][0][0] <= cor[1][0] <= grid_pos[i][1][0] and grid_pos[i][0][1] <= cor[1][1] <= grid_pos[i][2][1]):
                cen_edge[i].append(edge)
                weight[i] += 1
            else:  # 两个端点都不在方格内
                if cor[0][0] - cor[1][0] == 0:  # 平行y轴

                    if grid_pos[i][0][0] <= cor[0][0] <= grid_pos[i][1][0]:
                        # 检查网格单元格的y坐标是否在线段的y坐标范围内
                        miny = min(cor[0][1], cor[1][1])
                        maxy = max(cor[0][1], cor[1][1])
                        if miny <= grid_pos[i][0][1] and maxy >= grid_pos[i][2][1]:
                            cen_edge[i].append(edge)
                            weight[i] += 1



                elif cor[0][1] - cor[1][1] == 0:  # 平行x轴

                    if grid_pos[i][0][1] <= cor[0][1] <= grid_pos[i][2][1]:
                        # 检查网格单元格的y坐标是否在线段的y坐标范围内
                        minx = min(cor[0][0], cor[1][0])
                        maxx = max(cor[0][0], cor[1][0])
                        if minx <= grid_pos[i][0][0] and maxx >= grid_pos[i][1][0]:
                            cen_edge[i].append(edge)
                            weight[i] += 1

                else:

                    slope = (cor[1][1] - cor[0][1]) / (cor[1][0] - cor[0][0])  # 斜率
                    intercept = cor[0][1] - slope * cor[0][0]  # 截距
                    # print(slope)
                    # print(intercept)

                    ver_x1 = grid_pos[i][0][0]
                    ver_y1 = slope * ver_x1 + intercept

                    ver_x2 = grid_pos[i][1][0]
                    ver_y2 = slope * ver_x2 + intercept

                    ver_y3 = grid_pos[i][0][1]
                    ver_x3 = (ver_y3 - intercept)/slope

                    ver_y4 = grid_pos[i][2][1]
                    ver_x4 = (ver_y4 - intercept)/slope


                    min_x_cor = min(cor[0][0], cor[1][0])
                    max_x_cor = max(cor[0][0], cor[1][0])
                    min_y_cor = min(cor[0][1], cor[1][1])
                    max_y_cor = max(cor[0][1], cor[1][1])

                    min_x_squa = min(grid_pos[i][0][0], grid_pos[i][1][0])
                    max_x_squa = max(grid_pos[i][0][0], grid_pos[i][1][0])
                    min_y_squa = min(grid_pos[i][0][1], grid_pos[i][2][1])
                    max_y_squa = max(grid_pos[i][0][1], grid_pos[i][2][1])

                    # 使用 is_close 函数来比较浮点数
                    points = [(ver_x1, ver_y1), (ver_x2, ver_y2), (ver_x3, ver_y3), (ver_x4, ver_y4)]
                    for point in points:
                        if is_close(point[0], min_x_cor, epsilon) or is_close(point[0], max_x_cor, epsilon) or (min_x_cor < point[0] < max_x_cor):
                            if is_close(point[1], min_y_cor, epsilon) or is_close(point[1], max_y_cor, epsilon) or (min_y_cor < point[1] < max_y_cor):
                                if is_close(point[0], min_x_squa, epsilon) or is_close(point[0], max_x_squa,epsilon) or (min_x_squa < point[0] < max_x_squa):
                                    if is_close(point[1], min_y_squa, epsilon) or is_close(point[1], max_y_squa,epsilon) or (min_y_squa < point[1] < max_y_squa):
                                        cen_edge[i].append(edge)
                                        weight[i] += 1
                                        break

    return cen_edge, weight




def ExhaustiveSearch(G, Grid, cen_edge, top_num):  # 计算 1 2 3 4 5的所有最优组合

    original_largest_cc = len(max(nx.connected_components(G), key=len))
    grid_index = list(Grid.keys())
    all_optimal_sets=[]
    for top in range(1, top_num+1):
        delta_LCC = []
        all_possible = list(list(i) for i in combinations(grid_index, top))
        for items in all_possible:
            attack_edge = []
            for item in items:
                if len(cen_edge[item]) > 0:
                    attack_edge.extend(cen_edge[item])
            attack_edge = list(set(attack_edge))
            G_copy = copy.deepcopy(G)
            G_copy.remove_edges_from(attack_edge)
            residual_largest_cc = len(max(nx.connected_components(G_copy), key=len))
            delta = original_largest_cc - residual_largest_cc
            delta_LCC.append(delta)

            max_value = max(delta_LCC)  # 找到最大值
            indices = [i for i, x in enumerate(delta_LCC) if x == max_value]  # 找到最大值的所有索引
            optimal_sets = [all_possible[i] for i in indices]
        all_optimal_sets.extend(optimal_set for optimal_set in optimal_sets)

    return all_optimal_sets


# def ExhaustiveSearch(G, Grid, cen_edge, top_num): # 仅计算5的所有最优组合
#
#     original_largest_cc = len(max(nx.connected_components(G), key=len))
#     grid_index = list(Grid.keys())
#     delta_LCC = []
#     all_possible = list(list(i) for i in combinations(grid_index, top_num))
#     for items in all_possible:
#         attack_edge = []
#         for item in items:
#             if len(cen_edge[item]) > 0:
#                 attack_edge.extend(cen_edge[item])
#         attack_edge = list(set(attack_edge))
#         G_copy = copy.deepcopy(G)
#         G_copy.remove_edges_from(attack_edge)
#         residual_largest_cc = len(max(nx.connected_components(G_copy), key=len))
#         delta = original_largest_cc - residual_largest_cc
#         delta_LCC.append(delta)
#     # print("剩余连通片与初始连通片的最大差值（效果最好）", max(delta_LCC))
#
#     max_value = max(delta_LCC)  # 找到最大值
#     indices = [i for i, x in enumerate(delta_LCC) if x == max_value]  # 找到最大值的所有索引
#     optimal_sets = [all_possible[i] for i in indices]
#     # print("最优组合：", optimal_sets)
#
#     return optimal_sets



def Initial_Lable(optimal_sets):

    # Initial Training Score
    grids = list(range(pow(5, 2)))
    num_grids = len(grids)

    frequency_score = [0] * num_grids
    for optimal_set in optimal_sets:
        for item in optimal_set:
            frequency_score[grids.index(item)] += 1          # 每个节点出现在最优TAS中的次数。 对于每个最优集合中的节点，通过 nodes.index(item) 找到其在节点列表 nodes 中的索引，并将对应位置的 initial_score 值加1

    frequency_score = np.array(frequency_score).reshape(-1,1)  # reshape(-1,1) 转成列向量
    # print("出现频率：", frequency_score)
    frequency_score = frequency_score / np.sum(frequency_score)  # 分数归一化到范围 [0, 1]
    frequency_score = frequency_score.reshape(-1)
    # print("频率分数：", frequency_score)
    return frequency_score



# def Label_Normalied_Rank(score,grid_weight_dict):
#
#
#     grids = list(range(pow(5, 2)))
#     score = score.reshape(-1)              # 将数组转化为一维数组，即重组为一行
#     # print("分数：", score)
#     out = np.sort(score)                   # 从小到大排序后的分数
#     # print("从小到大排序后的分数：",out)
#     scorecount = collections.Counter(out)  # 统计out数组中每个分数出现的次数，并将结果存储在scorecount字典中。
#     # print("每个分数出现的次数：", scorecount)
#
#     rank = np.argsort(score)               # np.argsort()返回从小到大排序的索引值，即原始分数数组中元素的排名。
#     # print("排名：", rank)
#     norm_label = np.empty(rank.shape)      # 创建一个形状与列表rank相同，注意元素都未初始化
#     # print(norm_label)
#     i = 0
#     while i < len(rank):
#         if scorecount[out[i]] == 1 and score[rank[i]] == out[i]: # 检查当前分数是否只出现了一次，并且当前分数与排序后的分数数组中对应位置的分数相等
#             norm_label[rank[i]] = float(i+1) / float(len(rank))
#             i = i + 1
#
#         elif scorecount[out[i]] != 1:
#             SameScoreNodes = [grids[rank[j]] for j in range(i, i + scorecount[out[i]])]       # 获取当前分数出现多次时的节点列表
#             SameNodesDegree = np.array([grid_weight_dict[s] for s in SameScoreNodes])         # 获取具有相同分数的节点的度信息。
#             DegreeRank = np.argsort(SameNodesDegree)                                          # 对节点按度进行排序，得到排序的索引。
#             for j in range(len(DegreeRank)):
#                 norm_label[grids.index(SameScoreNodes[DegreeRank[j]])] = float(i + 1) / float(len(rank))
#                 i = i + 1
#
#     norm_label = np.array(norm_label).reshape(-1)
#     # print("标准化标签：", norm_label)
#     return norm_label





def PrepareTrainData(graph_type_list, num_graph):

    Adj = []
    Feature = []
    Label_Frequency = []

    for graph_type in graph_type_list:
        DATA_PATH = os.path.join(os.getcwd(), 'data', 'train', '30_50_' + graph_type + '_graph')
        for id in range(num_graph):

            File_Path1 = os.path.join(DATA_PATH, 'GridAdjFile', 'train_adj_'+str(id)+'.npy')
            if os.path.isfile(File_Path1):
                adj_id = pickle_read(File_Path1)
                Adj.append(adj_id)

            File_Path2 = os.path.join(DATA_PATH, 'GridFeatureMatrixFile', 'train_grid_feature_matrix_'+str(id)+'.npy')
            if os.path.isfile(File_Path2):
                feature_id = pickle_read(File_Path2)
                Feature.append(feature_id)

            File_Path3 = os.path.join(DATA_PATH, 'FrequencyScoreFile', 'train_label_' + str(id) + '.npy')
            if os.path.isfile(File_Path3):
                label_id = pickle_read(File_Path3)
                Label_Frequency.append(label_id)
                # Label_Frequency.append(label_id[:, None])

    SAVE_PATH = os.path.join(os.getcwd(), 'data', 'train', 'dataset')
    os.makedirs(SAVE_PATH, exist_ok=True)
    pickle_save(os.path.join(SAVE_PATH, 'train_dataset_grid_adj.npy'), Adj)
    pickle_save(os.path.join(SAVE_PATH, 'train_dataset_feature.npy'), Feature)
    pickle_save(os.path.join(SAVE_PATH, 'train_dataset_label.npy'), Label_Frequency)




if __name__ == '__main__':

    # start_time = time.time()  # 记录开始时间
    Synthetic_Type = ['BA']  # 'BA', 'ER', 'WS'
    # Network_Layout = ['KKL']  # 'KKL', 'SPEL' 'RL' 'CL' 'SPIL'

    # for type in Synthetic_Type:  # 每种模型生成1000个网络
    #     for g_layout in Network_Layout:
            # if g_layout == 'RL':
            #     initnum = 0
            #     endnum = 1000
            # if g_layout == 'KKL':
            #     initnum = 5000
            #     endnum = 7500
            # if g_layout == 'SPEL':
            #     initnum = 2000
            #     endnum = 3000
    #         # if g_layout == 'CL':
    #         #     initnum = 150
    #         #     endnum = 200
    #         # if g_layout == 'SPIL':
    #         #     initnum = 200
    #         #     endnum = 250
    #
            # for id in range(initnum, endnum):
            #         GenerateTrainData(id, type, g_layout)


    # end_time = time.time()
    # print("程序运行时间：", end_time - start_time, "秒")

    #
    num_graph = 25000
    PrepareTrainData(Synthetic_Type, num_graph)




