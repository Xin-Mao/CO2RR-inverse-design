import time
import argparse
import torch
import collections
import warnings

import numpy as np
import bird_swarm_opt
from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch

from eval_utils import load_model
from pymatgen.core import Lattice, Structure, Molecule
from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.models import dimenet_plus_plus
from catalyst import build_surface_with_absorbate
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Poscar

warnings.filterwarnings('ignore')

def bsa_optimization(model,
                 ld_kwargs,
                 data_loader,
                 num_starting_points=100,
                 num_steps=1000,
                 interval=10,
                 lr=1e-3):
    if data_loader is not None:
        batch = next(iter(data_loader)).to(model.device)
        _, _, z = model.encode(batch)
        z = z[:num_starting_points].detach().clone()
    else:
        z = torch.randn(num_starting_points,
                        model.hparams.hidden_dim,
                        device=model.device)
    model.freeze()
    bsa = bird_swarm_opt.Bird_swarm_opt(opt_matrix=z,
                                        cdvae_model=model,
                                        ld_kwargs=ld_kwargs,
                                        interval=interval,
                                        ck_path='best_checkpoint_dimenet.pt')
    conti_iter = int(torch.load("index.pt"))
    fMin, bestIndex, bestX, b2 = bsa.search(M=num_steps,
                                            conti=True,
                                            conti_iteration=conti_iter)
    # fMin, bestIndex, bestX, b2 = bsa.search(M=num_steps)
    np.save('fMin.npy', fMin)
    np.save('bestIndex.npy', bestIndex)
    np.save('bestX.npy', bestX)
    np.save('b2.npy', b2)
    all_crystals = bsa.all_crystals
    return {
        k: torch.cat([d[k] for d in all_crystals]).unsqueeze(0)
        for k in
        ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']
    }


def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, test_loader, cfg = load_model(
        model_path,
        load_data= True)
    ld_kwargs = SimpleNamespace(n_step_each=args.n_step_each,
                                step_lr=args.step_lr,
                                min_sigma=args.min_sigma,
                                save_traj=args.save_traj,
                                disable_bar=args.disable_bar)

    if torch.cuda.is_available():
        model.to('cuda')

    start_time = time.time()
    if args.start_from == 'data':
        loader = test_loader
    else:
        loader = None
    optimized_crystals = bsa_optimization(model, ld_kwargs, loader)
    optimized_crystals.update({
        'eval_setting': args,
        'time': time.time() - start_time
    })

    if args.label == '':
        gen_out_name = 'eval_opt.pt'
    else:
        gen_out_name = f'eval_opt_{args.label}.pt'
    torch.save(optimized_crystals, model_path / gen_out_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--n_step_each', default=100, type=int)
    parser.add_argument('--step_lr', default=1e-4, type=float)
    parser.add_argument('--min_sigma', default=0, type=float)
    parser.add_argument('--save_traj', default=False, type=bool)
    parser.add_argument('--disable_bar', default=False, type=bool)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--num_batches_to_samples', default=200, type=int)
    parser.add_argument('--start_from', default='no', type=str)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--force_num_atoms', action='store_true')
    parser.add_argument('--force_atom_types', action='store_true')
    parser.add_argument('--down_sample_traj_step', default=10, type=int)
    parser.add_argument('--label', default='')

    args = parser.parse_args()

    main(args)
