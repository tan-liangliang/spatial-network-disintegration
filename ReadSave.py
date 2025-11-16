import pickle
import os
import networkx as nx



def pickle_read(path):
    """从指定路径读取数据并返回"""
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def pickle_save(path, data):
    """将数据保存到指定路径的文件中"""
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def Save_Graph_GML(G, DATA_DIR_PATH, id):
    GraphFile = os.path.join(DATA_DIR_PATH, 'SourceGraphFile')
    os.makedirs(GraphFile, exist_ok=True)
    File = os.path.join(GraphFile, str(id)+'.gml')
    nx.write_gml(G, File)


def Save_Graph_Coordinate(Coordinate, DATA_DIR_PATH, id):
    """存储图坐标"""
    GraphFile = os.path.join(DATA_DIR_PATH, 'GraphCoordinateFile')
    os.makedirs(GraphFile, exist_ok=True)
    # File = os.path.join(GraphFile, str(id) + '.gpickle')
    File = os.path.join(GraphFile, str(id) + '.npy')
    # nx.write_gpickle(Coordinate, File)
    pickle_save(File, Coordinate)


def Save_Graph_Adj(adj, DATA_DIR_PATH, id):
    Graph_Adj_File = os.path.join(DATA_DIR_PATH, 'GraphAdjFile')
    os.makedirs(Graph_Adj_File, exist_ok=True)
    File = os.path.join(Graph_Adj_File, 'train_adj_'+str(id)+'.npy')
    pickle_save(File, adj)




def Save_Grid_Adj(adj, DATA_DIR_PATH, id):
    Grid_Adj_File = os.path.join(DATA_DIR_PATH, 'GridAdjFile')
    os.makedirs(Grid_Adj_File, exist_ok=True)
    File = os.path.join(Grid_Adj_File, 'train_adj_'+str(id)+'.npy')
    pickle_save(File, adj)




def Save_FeatureMatrix(FeatureMatrix, DATA_DIR_PATH, id):
    Graph_Feature_File = os.path.join(DATA_DIR_PATH, 'FeatureMatrixFile')
    os.makedirs(Graph_Feature_File, exist_ok=True)
    File = os.path.join(Graph_Feature_File, 'train_feature_'+str(id)+'.npy')
    pickle_save(File, FeatureMatrix)


def Save_OptimalSets(OptimalSets, DATA_DIR_PATH, id):
    """创建路径，将最优攻击节点集.npy文件存储至目标文件夹OptimalSetsFile"""
    OptimalFile = os.path.join(DATA_DIR_PATH, 'OptimalSetsFile')
    os.makedirs(OptimalFile, exist_ok=True)
    File = os.path.join(OptimalFile, 'train_optimal_sets_'+str(id)+'.npy')
    pickle_save(File, OptimalSets)


def Save_SquareWeights(weight, DATA_DIR_PATH, id):
    """创建路径，将最优攻击节点集.npy文件存储至目标文件夹OptimalSetsFile"""
    SquareWeightsFile = os.path.join(DATA_DIR_PATH, 'SquareWeightsFile')
    os.makedirs(SquareWeightsFile, exist_ok=True)
    File = os.path.join(SquareWeightsFile, 'train_square_weights_'+str(id)+'.npy')
    pickle_save(File, weight)

def Save_FrequencyLabel(lable, DATA_DIR_PATH, id):

    LabelFile = os.path.join(DATA_DIR_PATH, 'FrequencyScoreFile')
    os.makedirs(LabelFile, exist_ok=True)
    File = os.path.join(LabelFile, 'train_label_' + str(id) + '.npy')
    pickle_save(File, lable)

def Save_NormRankLabel(lable, DATA_DIR_PATH, id):
    """创建路径，将标准化排名标签.npy文件存储至目标文件夹OptimalSetsFile"""
    LabelFile = os.path.join(DATA_DIR_PATH, 'NormRankLabelFile')
    os.makedirs(LabelFile, exist_ok=True)
    File = os.path.join(LabelFile, 'train_label_' + str(id) + '.npy')
    pickle_save(File, lable)


def Save_GridFeatureMatrix(featurematrix, DATA_DIR_PATH, id):
    """创建路径，将标准化排名标签.npy文件存储至目标文件夹OptimalSetsFile"""
    LabelFile = os.path.join(DATA_DIR_PATH, 'GridFeatureMatrixFile')
    os.makedirs(LabelFile, exist_ok=True)
    File = os.path.join(LabelFile, 'train_grid_feature_matrix_' + str(id) + '.npy')
    pickle_save(File, featurematrix)