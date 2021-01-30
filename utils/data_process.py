import numpy as np
import scipy.sparse as sp
import torch
import collections
import sys, os
import pickle as pkl
import networkx as nx
from multiprocessing import Pool
from itertools import product
#from torch_geometric.datasets import CitationFull, Planetoid, Coauthor

folder = 'dataset/'
threshold = 5.0

def normalize_adj(adj, norm_type=1, iden=False):
    # 1: mean norm, 2: spectral norm
    # add the diag into adj, namely, the self-connection. then normalization
    if iden:
        adj = adj + np.eye(adj.shape[0])       # self-loop
    if norm_type==1:
        D = np.sum(adj, axis=1)
        adjNor = adj / D
        adjNor[np.isinf(adjNor)] = 0.
    else:
        adj[adj > 0.0] = 1.0
        D_ = np.diag(np.power(np.sum(adj, axis=1), -0.5)) 
        adjNor = np.dot(np.dot(D_, adj), D_)
    
    return adjNor, adj 


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)    
    mx = r_mat_inv.dot(mx)

    return mx


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    
    if sp.issparse(features):
        return features.todense()
    else:
        return features


def convert_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def convert_to_torch_tensor(features, adj, tail_adj, labels, idx_train, idx_val, idx_test):
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    adj = convert_sparse_tensor(adj)             #+ sp.eye(adj.shape[0]))
    tail_adj = convert_sparse_tensor(tail_adj)   #+ sp.eye(tail_adj.shape[0])    
    iden = sp.eye(adj.shape[0])
    iden = convert_sparse_tensor(iden)

    return features, adj, tail_adj, iden, labels, idx_train, idx_val, idx_test


def link_dropout(adj, idx, seed=0):
    
    #np.random.seed(seed)

    tail_adj = adj.copy()
    num_links = np.random.randint(threshold, size=idx.shape[0]) 
    num_links += 1

    for i in range(idx.shape[0]):
        index = tail_adj[idx[i]].nonzero()[1]
        new_idx = np.random.choice(index, num_links[i], replace=False)
        tail_adj[idx[i]] = 0.0
        for j in new_idx:
            tail_adj[idx[i], j] = 1.0
    return tail_adj


# split head vs tail nodes
def split_nodes(adj):

    num_links = np.sum(adj, axis=1)
    idx_train = np.where(num_links > threshold)[0]
        
    idx_valtest = np.where(num_links <= threshold)[0]
    np.random.shuffle(idx_valtest)

    p = int(idx_valtest.shape[0] / 3)
    idx_val = idx_valtest[:p]
    idx_test = idx_valtest[p:]

    return idx_train, idx_val, idx_test
   

def process_email(path):

    load_features = np.genfromtxt("{}graph.embeddings".format(path), skip_header=1, dtype=np.float32)
    idx = load_features[load_features[:,0].argsort()] # sort regards to ascending index
    features = idx[:,1:]

    load_labels = np.genfromtxt("{}labels.txt".format(path), dtype=np.int32)
    labels = encode_onehot(load_labels[:,1])

    edges = np.genfromtxt("{}edges.txt".format(path), dtype=np.int32)    
    #print(edges.shape[0])
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(idx.shape[0], idx.shape[0]),
                        dtype=np.float32)
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj.tolil()    
    
    # remove self connection
    for i in range(adj.shape[0]):
        adj[i, i] = 0.

    # label head tail for train/test
    idx_train, idx_val, idx_test = split_nodes(adj)
    return features, adj, np.argmax(labels,1), (idx_train, idx_val, idx_test)


def process_wiki(path):

    load_features = np.genfromtxt("{}wiki.embeddings".format(path), skip_header=1, dtype=np.float32)
    idx = load_features[load_features[:,0].argsort()] # sort regards to ascending index
    features = idx[:,1:]

    load_labels = np.genfromtxt("{}category.txt".format(path), dtype=np.int32)
    labels = encode_onehot(load_labels[:,1])

    edges = np.genfromtxt("{}edgelist.txt".format(path), dtype=np.int32)    
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(idx.shape[0], idx.shape[0]),
                        dtype=np.float32)

    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    #preprocess for redundant edges
    adj = adj.tolil()
    ind = np.where(adj.todense() > 1.0)
    for i in range(ind[0].shape[0]):
        adj[ind[0][i], ind[1][i]] = 1.
    
    for i in range(adj.shape[0]):
        adj[i, i] = 0.

    #label head tail for train/test
    idx_train, idx_val, idx_test = split_nodes(adj)
    return features, adj, labels, (idx_train, idx_val, idx_test)


def process_flickr(path):
    
    if os.path.exists(path + 'feat.npy'):
        with open(path + 'feat.npy', 'rb') as f:
            idx = np.load(f)
            features = np.load(f)
            labels = np.load(f, allow_pickle=True)
    else:
        load_features = np.genfromtxt("{}graph.embeddings".format(path), skip_header=1, dtype=np.float32)
        idx = load_features[load_features[:,0].argsort()] # sort regards to ascending index
        features = idx[:,1:]

        labels = np.genfromtxt("{}group-edges.csv".format(path),delimiter=',', dtype=np.int32)
        labels -= 1
        max_class = np.max(labels[:,1]) + 1
        labels = sp.coo_matrix((np.ones(labels.shape[0]), (labels[:, 0], labels[:, 1])),
                            shape=(labels.shape[0], max_class),
                            dtype=np.int32)     
        with open(path + 'feat.npy', 'wb') as f:
            np.save(f, idx)
            np.save(f, features)
            np.save(f, labels)

    print('Done loading')

    labels = np.genfromtxt("{}group-edges.csv".format(path),delimiter=',', dtype=np.int32)
    labels -= 1
    max_class = np.max(labels[:,1]) + 1
    labels = sp.coo_matrix((np.ones(labels.shape[0]), (labels[:, 0], labels[:, 1])),
                        shape=(labels.shape[0], max_class),
                        dtype=np.int32)

    labels = labels.todense()

    edges = np.genfromtxt("{}edges.csv".format(path), delimiter=',', dtype=np.int32)    
    edges -= 1 
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(idx.shape[0], idx.shape[0]),
                        dtype=np.float32)
    
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj.tolil()    
  
    for i in range(adj.shape[0]):
        adj[i, i] = 0.

    # label head tail for train/test
    idx_train, idx_val,  idx_test = split_nodes(adj)   
    return features, adj, labels, (idx_train, idx_val, idx_test)



def process_chameleon(path):

    with open("{}node_feature_label.txt".format(path), 'rb') as f:
        clean_lines = (line.replace(b'\t',b',') for line in f)
        load_features = np.genfromtxt(clean_lines, skip_header=1, dtype=np.float32, delimiter=',')
    
    idx = load_features[load_features[:,0].argsort()] # sort regards to ascending index
    labels = encode_onehot(idx[:,-1])
    features = idx[:,1:-1]

    edges = np.genfromtxt("{}graph_edges.txt".format(path), dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(idx.shape[0], idx.shape[0]),
                        dtype=np.float32)
    
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj.tolil()
    
    for i in range(adj.shape[0]):
        adj[i, i] = 0.

    #label head tail for train/test
    idx_train, idx_val, idx_test = split_nodes(adj)
    tail_adj = adj.copy()
    tail_adj = link_dropout(tail_adj, idx_train)

    features = preprocess_features(features)
    return features, adj, labels, (idx_train, idx_val, idx_test)



def process_webkb(path):
    with open("{}node_feature_label.txt".format(path), 'rb') as f:
        clean_lines = (line.replace(b'\t',b',') for line in f)
        load_features = np.genfromtxt(clean_lines, skip_header=1, dtype=np.float32, delimiter=',')
    
    idx = load_features[load_features[:,0].argsort()] # sort regards to ascending index
    labels = encode_onehot(idx[:,-1])
    features = idx[:,1:-1]

    edges = np.genfromtxt("{}graph_edges.txt".format(path), dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(idx.shape[0], idx.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj.tolil()
    
    for i in range(adj.shape[0]):
        adj[i, i] = 0.

    #label head tail for train/test
    idx_train, idx_val, idx_test = split_nodes(adj)
    features = preprocess_features(features)
    return features, adj, labels, (idx_train, idx_val, idx_test)


def process_actor(path):
    
    num_feat = 931
    def to_array(feat):
        new_feat = np.zeros(num_feat, dtype=float)
        for i in feat:
            new_feat[int(i)-1] = 1.
        return new_feat

    features = []
    labels = []

    with open("{}node_feature_label.txt".format(path), 'r') as f:
        f.readline()
        for line in f:
            idx, feat, label = line.strip().split('\t')
            feat = [n for n in feat.split(',')]
            
            labels.append(label)
            feat = to_array(feat)
            features.append(feat)        

    features = np.asarray(features)
    labels = encode_onehot(labels)
    labels = np.asarray(labels, dtype=int)

    edges = np.genfromtxt("{}graph_edges.txt".format(path), dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    
    adj = adj.tolil()
    ind = np.where(adj.todense() > 1.0)
    for i in range(ind[0].shape[0]):
        adj[ind[0][i], ind[1][i]] = 1.

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj.tolil()

    for i in range(adj.shape[0]):
        adj[i, i] = 0.

    #label head tail for train/test
    idx_train, idx_val, idx_test = split_nodes(adj)
    print(idx_train.shape, idx_val.shape, idx_test.shape)

    features = preprocess_features(features)
    return features, adj, labels, (idx_train, idx_val, idx_test)


def process_cs(path):
    # citation edge file is symmetric already, preprocess for redundant edges

    if os.path.exists(path + 'feat.npy') and os.path.exists(path + 'adj.npz'):
        with open(path + 'feat.npy', 'rb') as f:
            idx = np.load(f)
            features = np.load(f)
            labels = np.load(f)
        adj = sp.load_npz(path + 'adj.npz')
    else:
        load_features = np.genfromtxt("{}graph.node".format(path), skip_header=1, dtype=np.float32)
        idx = load_features[load_features[:,0].argsort()] # sort regards to ascending index
        labels = encode_onehot(idx[:,1])
        features = idx[:,2:]

        #write to npy file
        with open(path + 'feat.npy', 'wb') as f:
            np.save(f, idx)
            np.save(f, features)
            np.save(f, labels)
        
        edges = np.genfromtxt("{}graph.edge".format(path), dtype=np.int32)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(idx.shape[0], idx.shape[0]),
                        dtype=np.float32)
        sp.save_npz(path + 'adj', adj)
    
    print('Done loading')

    adj = adj.tolil()
    #preprocess
    ind = np.where(adj.todense() > 1.0)
    for i in range(ind[0].shape[0]):
        adj[ind[0][i], ind[1][i]] = 1.
    
    for i in range(adj.shape[0]):
        adj[i, i] = 0.

    # label head tail for train/test
    idx_train, idx_val, idx_test = split_nodes(adj)
    features = preprocess_features(features)

    return features, adj, labels, (idx_train, idx_val, idx_test)


def process_physics(path):
    if os.path.exists(path + 'feat.npy') and os.path.exists(path + 'adj.npz'):
        with open(path + 'feat.npy', 'rb') as f:
            features = np.load(f)
            labels = np.load(f)
        adj = sp.load_npz(path + 'adj.npz') 

    else:
        data = Coauthor(path, name='Physics')[0]
        features = data.x
        labels = data.y

        edges = np.transpose(data.edge_index)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(data.x.shape[0], data.x.shape[0]),
                            dtype=np.float32)
        with open(path + 'feat.npy', 'wb') as f:
            np.save(f, features)
            np.save(f, labels)
        sp.save_npz(path + 'adj', adj)

    adj = adj.tolil()
    idx_train, idx_val, idx_test = split_nodes(adj)
    print(idx_train.shape, idx_val.shape, idx_test.shape)

    return features, adj, labels, (idx_train, idx_val, idx_test)



def process_cora(path):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}ind.cora.{}".format(path, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}ind.cora.test.index".format(path))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]


    #label head tail for train/test
    idx_train, idx_val, idx_test = split_nodes(adj.todense())
    features = preprocess_features(features)

    return features, adj.tolil(), np.argmax(labels,1), (idx_train, idx_val, idx_test)


def process_citeseer(path):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}ind.citeseer.{}".format(path, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/ind.citeseer.test.index".format(path))
    test_idx_range = np.sort(test_idx_reorder)

    # Fix citeseer dataset (there are some isolated nodes in the graph)
    # Find isolated nodes, add them as zero-vecs into the right position
    test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    tx_extended[test_idx_range-min(test_idx_range), :] = tx
    tx = tx_extended
    ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    ty_extended[test_idx_range-min(test_idx_range), :] = ty
    ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    #label head tail for train/test
    idx_train, idx_val, idx_test = split_nodes(adj.todense())
    features = preprocess_features(features)  
    
    return features, adj.tolil(), np.argmax(labels,1), (idx_train, idx_val, idx_test)


def process_pubmed(path):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}ind.pubmed.{}".format(path, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}ind.pubmed.test.index".format(path))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

  
    adj = adj.tolil()

    #label head tail for train/test
    idx_train, idx_val, idx_test = split_nodes(adj)
    features = preprocess_features(features)  

    return features, adj, np.argmax(labels,1), (idx_train, idx_val, idx_test)



def process_corafull(path):
    
    if os.path.exists(path + 'feat.npy') and os.path.exists(path + 'adj.npz'):
        with open(path + 'feat.npy', 'rb') as f:
            features = np.load(f)
            labels = np.load(f)
        adj = sp.load_npz(path + 'adj.npz') 
    
    else:
        data = CitationFull(path, name='Cora')[0]
        features = data.x
        labels = data.y
        edges = np.transpose(data.edge_index)
        
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(data.x.shape[0], data.x.shape[0]),
                            dtype=np.float32)
        features = preprocess_features(features)

        with open(path + 'feat.npy', 'wb') as f:
            np.save(f, features)
            np.save(f, labels)
        sp.save_npz(path + 'adj', adj)
    
    adj = adj.tolil()
    idx_train, idx_val, idx_test = split_nodes(adj)
    
    return features, adj, np.argmax(labels,1), (idx_train, idx_val, idx_test)
    

def load_dataset(dataset, path=None):
    
    np.random.seed(0)
    if path == None:
        path = folder + dataset + '/'
    else:
        path = path + dataset + '/'

    DATASET = {
        'email': process_email,
        'wiki': process_wiki,
        'flickr': process_flickr,
        'chameleon': process_chameleon,
        'cs-citation': process_cs,
        'physics': process_physics,
        'cora': process_cora,
        'citeseer': process_citeseer,
        'pubmed': process_pubmed,
        'cora_full': process_corafull,
        'wisconsin': process_webkb,
        'cornell': process_webkb,
        'texas': process_webkb,
        'squirrel': process_webkb,
        'actor': process_actor
    }   

    if dataset not in DATASET:
        return ValueError('Dataset not available')
    else:
        return DATASET[dataset](path=path)



if __name__ == "__main__":
    _, adj, _, _ = load_dataset('actor', path='../dataset/')
        