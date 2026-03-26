import json
import numpy as np
import re
from pathlib import Path
import sys
import argparse
# from irff.reax import ReaxFF
from irff.mpnn import MPNN

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
parser.add_argument('--dataset_group', default='all', type=str, help="Training subset: all, bcc, fcc, or bcc_fcc (aliases: both, bcc+fcc).")

# Physics Control Flags (1 = Optimise / 0 = Freeze)
parser.add_argument('--t', default=1, type=int, help='Toggle for three-body (angle) term optimisation.')
parser.add_argument('--h', default=0, type=int, help='Toggle for hydrogen bond term optimisation.')
parser.add_argument('--a', default=1, type=int, help='Toggle for angle term supervision (general bending).')
parser.add_argument('--f', default=1, type=int, help='Toggle for four-body (torsion) term optimisation.')
parser.add_argument('--bo', default=1, type=int, help='Toggle for bond order/energy optimisation.')
parser.add_argument('--vdw', default=1, type=int, help='Toggle for van der Waals (non-bonded) energy optimisation.')

# Arguments are parsed from the system command line.
args = parser.parse_args(sys.argv[1:])

# Load training trajectory files into a dictionary
traj_dir = Path(args.train_dir)

# Check if the directory exists to prevent runtime errors if the path is wrong
if not traj_dir.exists():
    raise FileNotFoundError(f"Train directory not found at: {traj_dir}. Please run the dataset split pipeline first.")

dataset = {}
# Changed to rglob to safely catch any nested files within the train directory
for traj_file in traj_dir.rglob("*.traj"):
    system_name = traj_file.stem
    system_name = re.sub(r"[^\w.-]", "_", system_name)  
    dataset[system_name] = str(traj_file)

dataset_group = args.dataset_group.strip().lower()
if dataset_group in ('bcc+fcc', 'both'):
    dataset_group = 'bcc_fcc'
if dataset_group not in ('all', 'bcc', 'fcc', 'bcc_fcc'):
    raise ValueError(
        f"Unsupported dataset_group '{args.dataset_group}'. Use one of: all, bcc, fcc, bcc_fcc (or alias both / bcc+fcc)."
    )

if dataset_group != 'all':
    filtered_dataset = {}
    for system_name, traj_path in dataset.items():
        upper_name = system_name.upper()
        if dataset_group == 'bcc' and 'BCC' in upper_name:
            filtered_dataset[system_name] = traj_path
        elif dataset_group == 'fcc' and 'FCC' in upper_name:
            filtered_dataset[system_name] = traj_path
        elif dataset_group == 'bcc_fcc' and ('BCC' in upper_name or 'FCC' in upper_name):
            filtered_dataset[system_name] = traj_path
    dataset = filtered_dataset

if not dataset:
    raise ValueError(
        f"No training trajectories matched dataset_group '{args.dataset_group}' in {traj_dir}."
    )

print(f"Loaded {len(dataset)} training configurations from {traj_dir} (dataset_group={dataset_group}).")

# Define parameter constraints for optimisation
# Freeze mass and cutoff, along with bond-order correction terms (boc1-boc5)
cons = ['mass', 'cutoff', 'boc1', 'boc2', 'boc3', 'boc4', 'boc5']

# Additional parameter blocks are appended to the 'cons' list
# if the corresponding physical toggle is set to 0.

# Torsion/Four-body terms: Includes V1, V2, V3 and torsion coefficients 1-4.
if not args.f:
    cons += ['tor1', 'tor2', 'tor3', 'tor4', 'V1', 'V2', 'V3'] 

# Valency/Three-body terms: General valency parameters (val1 through val7).
if not args.t:
    cons += ['val1', 'val2', 'val3', 'val6', 'val7'] 

# Angle terms: Includes equilibrium angle (theta0) and bending force constants.
if not args.a:
    cons += ['theta0', 'val9', 'val10', 'val8', 'vale', 'valang', 'val4', 'val5'] 

# Van der Waals terms: Non-bonded radii (rvdw), well depth (Devdw), and scaling.
if not args.vdw:
    cons += ['vdw1', 'gammaw', 'rvdw', 'alfa', 'Devdw']

# Hydrogen Bonding: Parameters governing the rohb and Dehb attractive terms.
# (Defaulted to 0/frozen in argparse since Li-metal lacks H-bonds).
if not args.h:
    cons += ['rohb', 'Dehb', 'hb1', 'hb2']

# Bond-Order and Bond Energy: Includes dissociation energies (Desi, Depi, Depp)
# and bond radii for sigma, pi, and double-pi bonds.
if not args.bo:
    cons += [
        'Depi', 'Depp', 'Desi',
        'rosi', 'ropi', 'ropp',
        'bo1', 'bo2', 'bo3', 'bo4', 'bo5', 'bo6'
    ]

if __name__ == '__main__':
    # Bullet-proof fix for the __spec__ crash in IPython/interactive environments
    import __main__
    if not hasattr(__main__, '__spec__'):
        __main__.__spec__ = None

    # Initialise MPNN model for energy-only optimisation
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
              energy_term={'ecoul': False, 'eunder': False, 'eover' : False} # ADD 'eunder': False TO BYPASS THE NaN CRASH and eover as a contigency
              )
    # Execution / Training Loop
    print(f"Starting MPNN training for {args.step} iterations with learning rate {args.lr}...") 
    rn.run(learning_rate=args.lr, 
           step=args.step, 
           print_step=args.pr, 
           writelib=args.writelib)

    print("Training complete. Force field saved.")

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
