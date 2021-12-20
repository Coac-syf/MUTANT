import scipy.sparse as sp

from utils import BatchSlidingWindow, get_data, get_data_dim
from sklearn.metrics.pairwise import cosine_similarity as cos
import numpy as np
def construct_graph(features, topk):
    fname = 'tmp.txt'
    fname2 = 'tmp2.txt'
    f = open(fname, 'w')
    f2 = open(fname2, 'w')

    dist = np.corrcoef(features)
    inds = []
    negs = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        neg = np.argpartition(dist[i, :], (topk + 1))[: topk+1]
        inds.append(ind)
        negs.append(neg)

    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                f.write('{} {}\n'.format(i, vv))
    f.close()

    for i, v in enumerate(negs):
        for vv in v:
            if vv == i:
                pass
            else:
                f2.write('{} {}\n'.format(i, vv))
    f2.close()

def normalize(mx):#特征归一化函数
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def generate_knn(data):
    topk = 6
    construct_graph(data, topk)

def returnA(x):

    x = np.array(x).T
    generate_knn(x)
    featuregraph_path = 'tmp.txt'
    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)  # 加载文件中信息

    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(x.shape[0], x.shape[0]),
                         dtype=np.float32)  # 构造邻接矩阵，此时图为非对称矩阵
    fadj = fadj+sp.coo_matrix(np.eye(x.shape[0]))
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)  # 变为对称矩阵
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))
    nfadj=nfadj.A
    return nfadj

