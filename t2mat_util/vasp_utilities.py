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


def draw_absorption_from_vaspkit(save_name='Absorption.pdf',
                                 file_name='ABSORPTION.dat'):
    df = pd.read_csv(file_name, sep='\s+')
    df = df[(df['#energy'] >= 0) & (df['#energy'] <= 5)]
    mean_absorp = (df['xx(cm^-1)'] + df['yy(cm^-1)'] + df['zz(cm^-1)'] +
                   df['xy(cm^-1)'] + df['zx(cm^-1)'] + df['yz(cm^-1)']) / 6
    integral = integrate.trapz(mean_absorp, list(df['#energy']))
    plt.rcParams['font.size'] = 36
    plt.rcParams['font.sans-serif'] = ['Liberation Sans']
    plt.figure(figsize=(16, 16))
    plt.plot(list(df['#energy']), mean_absorp, linewidth=2)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Absorption coefficient $(cm^{-1})$')
    ax = plt.gca()
    bwith = 2
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.savefig(save_name, bbox_inches='tight', dpi=1200)
    plt.show()
    return integral
