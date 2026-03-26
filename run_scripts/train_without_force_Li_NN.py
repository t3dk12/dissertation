import json
import numpy as np
import re
from pathlib import Path
import sys
import argparse

## Modify JSON encoder to handle NumPy types
# Steps:
# 1. Save the original JSON encoder function.
# 2. Define a function that converts NumPy types to native Python types.
# 3. Overwrite the default JSON encoder with the modified function.
_original_default = json.JSONEncoder.default

def _numpy_default(self, obj):
    # Convert NumPy scalars to native Python scalars
    if isinstance(obj, np.generic):
        return obj.item()
    # Use original encoder for other types
    return _original_default(self, obj)

json.JSONEncoder.default = _numpy_default


def build_parser():
    parser = argparse.ArgumentParser(description='Train ReaxFF/MPNN without force (Energy-only Optimisation)')

    # Training Hyperparameters
    # 'step' dictates the total number of iterations for the optimiser.
    # 'lr' (learning rate) determines the step size taken towards the local minimum;
    # values that are too high can lead to divergence in the potential energy.
    parser.add_argument('--step', '-s', default=10000, type=int, help='Total training iterations (epochs).')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate for gradient updates.')
    parser.add_argument('--writelib', default=1000, type=int, help='Step interval for saving the parameter library to disk.')
    parser.add_argument('--pr', default=10, type=int, help='Step interval for console output/logging of loss metrics.')
    parser.add_argument('--batch', default=50, type=int, help='Number of molecular configurations processed per batch.')
    parser.add_argument('--ffield', default='ffield.json', type=str, help='Path to the initial ReaxFF/MPNN force-field file.')

    # Dataset location
    parser.add_argument('--train_dir', default='data/traj/train', type=str, help='Directory containing the training .traj files.')
    parser.add_argument('--loader-timeout-seconds', default=900, type=float, help='Per-file loader timeout in seconds before quarantine.')
    parser.add_argument('--loader-workers', default=None, type=int, help='Optional override for loader workers; leave unset to use all CPU cores.')

    # Physics Control Flags (1 = Optimise / 0 = Freeze)
    parser.add_argument('--t', default=1, type=int, help='Toggle for three-body (angle) term optimisation.')
    parser.add_argument('--h', default=0, type=int, help='Toggle for hydrogen bond term optimisation.')
    parser.add_argument('--a', default=1, type=int, help='Toggle for angle term supervision (general bending).')
    parser.add_argument('--f', default=1, type=int, help='Toggle for four-body (torsion) term optimisation.')
    parser.add_argument('--bo', default=1, type=int, help='Toggle for bond order/energy optimisation.')
    parser.add_argument('--vdw', default=1, type=int, help='Toggle for van der Waals (non-bonded) energy optimisation.')
    return parser


def load_dataset(train_dir):
    traj_dir = Path(train_dir)
    if not traj_dir.exists():
        raise FileNotFoundError(
            f"Train directory not found at: {traj_dir}. Please run the dataset split pipeline first."
        )

    dataset = {}
    for traj_file in traj_dir.rglob("*.traj"):
        system_name = traj_file.stem
        system_name = re.sub(r"[^\w.-]", "_", system_name)
        dataset[system_name] = str(traj_file)

    print(f"Loaded {len(dataset)} training configurations from {traj_dir}.")
    return dataset


def build_constraints(args):
    # Freeze mass and cutoff, along with bond-order correction terms (boc1-boc5)
    cons = ['mass', 'cutoff', 'boc1', 'boc2', 'boc3', 'boc4', 'boc5']

    # Additional parameter blocks are appended to the 'cons' list if the
    # corresponding physical toggle is set to 0.
    if not args.f:
        cons += ['tor1', 'tor2', 'tor3', 'tor4', 'V1', 'V2', 'V3']

    if not args.t:
        cons += ['val1', 'val2', 'val3', 'val6', 'val7']

    if not args.a:
        cons += ['theta0', 'val9', 'val10', 'val8', 'vale', 'valang', 'val4', 'val5']

    if not args.vdw:
        cons += ['vdw1', 'gammaw', 'rvdw', 'alfa', 'Devdw']

    if not args.h:
        cons += ['rohb', 'Dehb', 'hb1', 'hb2']

    if not args.bo:
        cons += [
            'Depi', 'Depp', 'Desi',
            'rosi', 'ropi', 'ropp',
            'bo1', 'bo2', 'bo3', 'bo4', 'bo5', 'bo6'
        ]
    return cons


def main(argv=None):
    from irff.mpnn import MPNN

    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    dataset = load_dataset(args.train_dir)
    cons = build_constraints(args)

    # Bullet-proof fix for the __spec__ crash in IPython/interactive environments
    import __main__
    if not hasattr(__main__, '__spec__'):
        __main__.__spec__ = None

    rn = MPNN(libfile=args.ffield,
              dataset=dataset,
              batch_size=args.batch,
              losFunc='n2',
              vdwcut=10.0,
              cons=cons,
              mf_layer=[9, 1],
              be_layer=[9, 1],
              EnergyFunction=1,
              MessageFunction=3,
              messages=1,
              energy_term={'ecoul': False, 'eunder': False, 'eover': False},
              loader_workers=args.loader_workers,
              loader_timeout_seconds=args.loader_timeout_seconds
              )

    print(f"Starting MPNN training for {args.step} iterations with learning rate {args.lr}...")
    try:
        rn.run(learning_rate=args.lr,
               step=args.step,
               print_step=args.pr,
               writelib=args.writelib)
    except RuntimeError as exc:
        msg = str(exc)
        if "cell length must lager than 2.0*r_cut" in msg:
            raise RuntimeError(
                msg
                + "\nHint: rebuild dataset with duplication enabled, e.g. "
                  "`bash setup_and_run.sh` (it now auto-checks undersized cells)."
            ) from exc
        raise

    print("Training complete. Force field saved.")


if __name__ == '__main__':
    main()

# # Initialise MPNN model for energy-only optimisation
# rn = MPNN(libfile=args.ffield,
#           dataset=dataset,
#           batch_size=args.batch,
#           losFunc='n2',          # L2 norm (MSE) for the loss function
#           vdwcut=10.0,           # Global cutoff for van der Waals interactions (Angstroms)
#           cons=cons,             # List of frozen parameters
#           # ReaxFF-nn specific arguments
#           mf_layer=[9, 1],       # Neural network architecture for Message Passing / Bond Order (Input: 9 features -> Output: 1)
#           be_layer=[9, 1],       # Neural network architecture for Bond Energy calculations
#           EnergyFunction=1,      # Turns on NN formulation for energy calculations
#           MessageFunction=3,     # Turns on NN formulation for bond-orders (message passing style)
#           messages=1             # Number of message passing iterations (hops) between atoms
#          )

# # Execution / Training Loop
# print(f"Starting MPNN training for {args.step} iterations with learning rate {args.lr}...")
# rn.run(learning_rate=args.lr, 
#        step=args.step, 
#        print_step=args.pr, 
#        writelib=args.writelib)
# print("Training complete. Force field saved.")
