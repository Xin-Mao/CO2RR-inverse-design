import os
import json
import torch
import yaml
import numpy as np
import pandas as pd
from ase import Atoms
import torch
import lmdb
import pickle
from scipy import integrate
from torch.nn.functional import cosine_similarity
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp import Poscar
from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.datasets import SinglePointLmdbDataset
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from torch_geometric.data import Batch
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.model_selection import train_test_split
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.transformations.standard_transformations import RotationTransformation
from graph_transformer import Graph_transformer
from sklearn.cluster import DBSCAN
from scipy.spatial import distance


def load_model(model_name, model_settings_path, checkpoint_path):
    with open(model_settings_path, 'r') as file:
        data = yaml.safe_load(file)
    model_settings = data['model']
    del model_settings['name']
    model_settings['num_targets'] = 1
    model = model_name(**model_settings)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'))
    ckpt_key_count = next(iter(checkpoint["state_dict"])).count("module")
    mod_key_count = next(iter(model.state_dict())).count("module")
    key_count_diff = mod_key_count - ckpt_key_count
    if key_count_diff > 0:
        new_dict = {
            key_count_diff * "module." + k: v
            for k, v in checkpoint["state_dict"].items()
        }
    elif key_count_diff < 0:
        new_dict = {
            k[len("module.") * abs(key_count_diff):]: v
            for k, v in checkpoint["state_dict"].items()
        }
    else:
        new_dict = checkpoint["state_dict"]
    model.load_state_dict(new_dict, strict=False)
    return model


def get_struct_data(lmdb_data):
    return Batch.from_data_list([i for i in lmdb_data])


def get_input_data(data_list):
    return Batch.from_data_list(data_list)


def calculate_clusters(points, eps=0.1, min_samples=2):
    # 使用 DBSCAN 进行聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    # 计算聚类簇的数量（忽略噪声点）
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # 计算每个簇的平均距离
    avg_distances = []
    for cluster_id in range(n_clusters):
        cluster_points = points[labels == cluster_id]
        cluster_center = np.mean(cluster_points, axis=0)
        distances = distance.cdist(cluster_points, [cluster_center])
        avg_distance = np.mean(distances)
        avg_distances.append(avg_distance)

    # 计算所有簇的平均距离
    total_avg_distance = np.mean(avg_distances)

    # 计算综合指标
    indicator = n_clusters / total_avg_distance

    return indicator


def create_contrastive_dataset(structures):
    dataset = []
    # 对于数据集中的每个晶体结构
    for i in range(len(structures)):
        anchor = structures[i]
        # 为这个锚点创建一个正样本
        rotation_angle = np.random.uniform(0, 360)  # 选择一个随机的旋转角度
        axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        rotation_axis = axes[np.random.randint(3)]
        rotation_transformation = RotationTransformation(
            rotation_axis, rotation_angle)
        positive = rotation_transformation.apply_transformation(anchor)
        # 为这个锚点创建一个负样本
        negative_index = np.random.choice(len(structures))  # 从数据集中随机选择一个负样本
        while negative_index == i:  # 确保负样本和锚点不是同一个结构
            negative_index = np.random.choice(len(structures))
        negative = structures[negative_index]
        # 将锚点、正样本和负样本添加到数据集中
        dataset.append((AseAtomsAdaptor.get_atoms(anchor),
                        AseAtomsAdaptor.get_atoms(positive),
                        AseAtomsAdaptor.get_atoms(negative)))
    return dataset


def info_nce_loss(x, j, k, temperature=0.1, similarity='cosine'):
    """
    Compute InfoNCE loss.

    Args:
        x: Tensor of shape [batch_size, feature_dim]. Model's output (anchor points).
        j: Tensor of shape [batch_size, feature_dim]. Positive samples.
        k: Tensor of shape [batch_size, feature_dim]. Negative samples.
        temperature: A hyperparameter.

    Returns:
        loss: InfoNCE loss.
    """
    # Compute similarity scores
    if similarity == 'cosine':
        positive_similarity = F.cosine_similarity(x, j, dim=-1) / temperature
        negative_similarity = F.cosine_similarity(x, k, dim=-1) / temperature
    elif similarity == 'dot':
        positive_similarity = torch.sum(x * j, dim=-1) / temperature
        negative_similarity = torch.sum(x * k, dim=-1) / temperature
    # Compute logits
    logits = torch.cat(
        [positive_similarity.unsqueeze(1),
         negative_similarity.unsqueeze(1)],
        dim=1)

    # Create labels (positives: 0, negatives: 1)
    labels = torch.zeros(logits.shape[0],
                         dtype=torch.long,
                         device=logits.device)

    # Compute loss
    loss = torch.nn.functional.cross_entropy(logits, labels)

    return loss


def write_lmdb(atoms, energy, lmdb_name, lmdb_path):
    atg = AtomsToGraphs()
    db = lmdb.open(
        f'{lmdb_path}/{lmdb_name}',
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    data_objects = atg.convert_all(atoms)
    idx = 0
    for dataobject, ene in zip(data_objects, energy):
        dataobject.sid = idx
        dataobject.y_relaxed = ene
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(dataobject,
                                                       protocol=-1))
        txn.commit()
        db.sync()
        idx += 1
    db.close()
    dataset = SinglePointLmdbDataset({"src": f'{lmdb_path}/{lmdb_name}'})
    db_train = lmdb.open(
        f"{lmdb_path}/train_{lmdb_name}",
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    db_val = lmdb.open(
        f"{lmdb_path}/val_{lmdb_name}",
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    db_test = lmdb.open(
        f"{lmdb_path}/test_{lmdb_name}_bk",
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    data_len = len(dataset)
    indexs = list(range(data_len))
    train_index, val_test_index = train_test_split(indexs,
                                                   test_size=0.1,
                                                   random_state=42)
    val_index, _ = train_test_split(val_test_index,
                                    test_size=0.5,
                                    random_state=42)
    idx_train = 0
    idx_val = 0
    idx_test = 0
    for i, dataob in enumerate(dataset):
        if i in train_index:
            txn_train = db_train.begin(write=True)
            txn_train.put(f"{idx_train}".encode("ascii"),
                          pickle.dumps(dataob, protocol=-1))
            txn_train.commit()
            db_train.sync()
            idx_train += 1
        elif i in val_index:
            txn_val = db_val.begin(write=True)
            txn_val.put(f"{idx_val}".encode("ascii"),
                        pickle.dumps(dataob, protocol=-1))
            txn_val.commit()
            db_val.sync()
            idx_val += 1
        else:
            txn_test = db_test.begin(write=True)
            txn_test.put(f"{idx_test}".encode("ascii"),
                         pickle.dumps(dataob, protocol=-1))
            txn_test.commit()
            db_test.sync()
            idx_test += 1
    dataset_test = SinglePointLmdbDataset(
        {"src": f"{lmdb_path}/test_{lmdb_name}_bk"})
    db_test = lmdb.open(
        f"{lmdb_path}/test_{lmdb_name}",
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    idx_test = 0
    for i, dataob in enumerate(dataset_test):
        dataob['sid'] = idx_test
        txn_test = db_test.begin(write=True)
        txn_test.put(f"{idx_test}".encode("ascii"),
                     pickle.dumps(dataob, protocol=-1))
        txn_test.commit()
        db_test.sync()
        idx_test += 1


def write_CL_lmdb(dataset, lmdb_name, lmdb_path):
    atg = AtomsToGraphs()
    structures = [
        Structure(dataob.cell, dataob.atomic_numbers, dataob.pos)
        for dataob in dataset
    ]
    contrastive_dataset = create_contrastive_dataset(structures)
    contrastive_dataset_ob = [
        atg.convert_all(list(atom_turple), disable_tqdm=True)
        for atom_turple in contrastive_dataset
    ]
    db_train = lmdb.open(
        f"{lmdb_path}/train_{lmdb_name}",
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    db_val = lmdb.open(
        f"{lmdb_path}/val_{lmdb_name}",
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    db_test = lmdb.open(
        f"{lmdb_path}/test_{lmdb_name}_bk",
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    data_len = len(dataset)
    indexs = list(range(data_len))
    train_index, val_test_index = train_test_split(indexs,
                                                   test_size=0.1,
                                                   random_state=42)
    val_index, _ = train_test_split(val_test_index,
                                    test_size=0.5,
                                    random_state=42)
    idx_train = 0
    idx_val = 0
    idx_test = 0
    for i, dataob in enumerate(dataset):
        if i in train_index:
            dataob.anchor = contrastive_dataset_ob[i][0]
            dataob.positive = contrastive_dataset_ob[i][1]
            dataob.negetive = contrastive_dataset_ob[i][2]
            txn_train = db_train.begin(write=True)
            txn_train.put(f"{idx_train}".encode("ascii"),
                          pickle.dumps(dataob, protocol=-1))
            txn_train.commit()
            db_train.sync()
            idx_train += 1
        elif i in val_index:
            dataob.anchor = contrastive_dataset_ob[i][0]
            dataob.positive = contrastive_dataset_ob[i][1]
            dataob.negetive = contrastive_dataset_ob[i][2]
            txn_val = db_val.begin(write=True)
            txn_val.put(f"{idx_val}".encode("ascii"),
                        pickle.dumps(dataob, protocol=-1))
            txn_val.commit()
            db_val.sync()
            idx_val += 1
        else:
            dataob.anchor = contrastive_dataset_ob[i][0]
            dataob.positive = contrastive_dataset_ob[i][1]
            dataob.negetive = contrastive_dataset_ob[i][2]
            txn_test = db_test.begin(write=True)
            txn_test.put(f"{idx_test}".encode("ascii"),
                         pickle.dumps(dataob, protocol=-1))
            txn_test.commit()
            db_test.sync()
            idx_test += 1
