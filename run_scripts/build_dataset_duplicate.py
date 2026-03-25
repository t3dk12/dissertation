"""
DFT to MLIP Dataset Script
Reads VASP output files, optionally enlarges small boxes,
splits the data into training and testing sets, and creates a CSV summary.
All comments and messages are in plain English.
"""

import argparse
import csv
import json
import random
import re
import shutil
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import numpy as np
from pathlib import Path
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator


# Get the cutoff radius from ffield.json or fallback
# Steps:
# 1. check for ffield.json
# 2. read json. read "rcut" (cutoff radius)
# 3. If rcut is a dictionary, use the largest value
# 4. If rcut is a single number, use it
# 5. If reading fails, use the fallback provided
# 6. Return the cutoff as a float
def get_cutoff_radius(ffield_path: Path, fallback_cutoff: float) -> float:
    # 1. check for ffield.json
    if ffield_path.exists():
        try:
            # 2. read json. read "rcut" (cutoff radius).
            with open(ffield_path, 'r') as f:
                data = json.load(f)
                if "rcut" in data:
                    rcut_value = data["rcut"]
                    print(f"Found rcut={rcut_value} in {ffield_path}")

                    # 3. If rcut is a dictionary, use the largest value
                    if isinstance(rcut_value, dict):
                        max_cutoff = max(float(v) for v in rcut_value.values())
                        print(f"Maximum cutoff: {max_cutoff} Å")
                        return max_cutoff
                    # 4. If rcut is a single number, use it
                    else:
                        return float(rcut_value)
        except Exception as e:
            # 5. If reading fails, use the fallback provided.
            print(f"Warning: Could not read cutoff from {ffield_path} ({e}).")

    # 6. Return the cutoff as a float
    print(f"Using fallback cutoff: {fallback_cutoff} Å")
    return fallback_cutoff


# --- NEW MULTIPROCESSING HELPER FUNCTION ---
# This function handles exactly one folder. We separated this from the main loop
# so that multiple CPU cores can run this function at the exact same time.
def process_single_folder(folder: Path, input_dir: Path, output_dir: Path, unused_folder: Path, minimum_box_width: float, enlarge_small_boxes: bool):
    outcar_file = folder / "OUTCAR"
    relative_path = folder.relative_to(input_dir)

    # remove slashes and join with underscores
    base_filename = "_".join(relative_path.parts)
    final_file_path = output_dir / f"{base_filename}.traj"

    # Start tracker entry for this specific folder
    status_info = {
        "Status": "Valid",
        "Duplicated": "N",
        "Notes": "Original file is valid."
    }

    # Skip files that were already processed 
    existing_supercells = list(output_dir.glob(f"{base_filename}_supercell_*.traj"))
    if final_file_path.exists() or existing_supercells:
        return base_filename, status_info

    trajectory_frames = None
    error_message = ""

    # Read OUTCAR
    if outcar_file.exists():
        try:
            trajectory_frames = read(outcar_file, index=":")
        except Exception as e:
            error_message = str(e)

    # If reading fails, copy original file to unused_data and mark as failed
    if not trajectory_frames:
        failed_filename = f"{base_filename}_OUTCAR"
        status_info = {
            "Status": "Failed",
            "Duplicated": "N",
            "Notes": f"Could not read OUTCAR file: {error_message}",
            "Failed_File": failed_filename
        }
        shutil.copy(outcar_file, unused_folder / failed_filename)
        print(f"Failed to read {base_filename} -> copied original to unused_data as {failed_filename}")
        return base_filename, status_info

    # If enlargement is enabled, calculate box widths and compare to required minimum width
    if enlarge_small_boxes:
        last_frame = trajectory_frames[-1]
        cell = last_frame.get_cell()
        volume = cell.volume

        # Calculate widths in each direction
        cell_widths = [
            volume / np.linalg.norm(np.cross(cell[1], cell[2])),
            volume / np.linalg.norm(np.cross(cell[2], cell[0])),
            volume / np.linalg.norm(np.cross(cell[0], cell[1])),
        ]
        repetitions_needed = [int(np.ceil(minimum_box_width / width)) for width in cell_widths]

        # Duplicate frames if any direction is too small
        if any(r > 1 for r in repetitions_needed):
            rep_x, rep_y, rep_z = repetitions_needed
            total_cells = rep_x * rep_y * rep_z
            supercell_tag = f"_supercell_{rep_x}x{rep_y}x{rep_z}"
            final_file_path = output_dir / f"{base_filename}{supercell_tag}.traj"

            # Save original small file to unused_data
            original_file_path = unused_folder / f"{base_filename}_original.traj"
            write(original_file_path, trajectory_frames)

            new_trajectory_frames = []

            for frame in trajectory_frames:
                # Repeat atoms to create supercell
                supercell_atoms = frame.repeat((rep_x, rep_y, rep_z))

                # store original properties
                original_energy = frame.get_potential_energy() if frame.calc else None
                original_forces = frame.get_forces() if frame.calc else None
                original_stress = frame.get_stress() if frame.calc else None

                # calculate new properties for the supercell
                new_energy = original_energy * total_cells if original_energy else None
                new_forces = np.tile(original_forces, (total_cells, 1)) if original_forces is not None else None
                new_stress = original_stress

                sp_calc = SinglePointCalculator(
                    supercell_atoms, energy=new_energy, forces=new_forces, stress=new_stress
                )
                supercell_atoms.calc = sp_calc
                new_trajectory_frames.append(supercell_atoms)

            # Save supercell trajectory
            write(final_file_path, new_trajectory_frames)
            status_info["Duplicated"] = "Y"
            status_info["Notes"] = (
                f"Original box too small, multiplied by [{rep_x}x{rep_y}x{rep_z}], "
                f"original moved to unused_data"
            )
            print(f"Enlarged {base_filename} to {final_file_path.name}")
            return base_filename, status_info
    else:
        status_info["Notes"] += " (enlargement turned off)"

    # Save normally if enlargement is off or box is large enough
    write(final_file_path, trajectory_frames)
    
    return base_filename, status_info


# Read and process VASP output files across multiple CPU cores
# Steps:
# 1. Print starting message
# 2. Make folders for output and unused data
# 3. Find all directories containing OUTCAR files
# 4. Initialise tracker dictionary to record status, duplication, and notes
# 5. Set up parallel processing (multiprocessing) to process folders simultaneously
# 6. Return tracker dictionary
def extract_and_process(input_dir: Path, output_dir: Path, cutoff_radius: float, enlarge_small_boxes: bool, max_workers: int):
    # 1. Print starting message
    if enlarge_small_boxes:
        print(f"--- Phase 1: Reading files and checking box size (cutoff {cutoff_radius} Å) using {max_workers} cores ---")
    else:
        print(f"--- Phase 1: Reading files (enlargement turned off) using {max_workers} cores ---")

    # 2. Make folders for output and unused data
    output_dir.mkdir(exist_ok=True)
    unused_folder = output_dir / "unused_data"
    unused_folder.mkdir(exist_ok=True)

    # 3. Find all directories containing OUTCAR files
    calculation_folders = [p.parent for p in input_dir.rglob("OUTCAR")]
    if not calculation_folders:
        print(f"No OUTCAR files found in {input_dir}. Exiting.")
        sys.exit(1)

    # 4. Initialise tracker dictionary to record status, duplication, and notes
    file_status_info = {}
    minimum_box_width = 2.0 * cutoff_radius  # minimum width required for safe periodic boundaries

    # 5. Set up parallel processing (multiprocessing)
    # We use functools.partial to freeze the static arguments so we only have to pass the folder to the worker
    worker_function = partial(
        process_single_folder,
        input_dir=input_dir,
        output_dir=output_dir,
        unused_folder=unused_folder,
        minimum_box_width=minimum_box_width,
        enlarge_small_boxes=enlarge_small_boxes
    )

    # Dispatch tasks to the CPU pool
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all folders to the executor
        futures = [executor.submit(worker_function, folder) for folder in calculation_folders]
        
        # As each worker finishes its folder, collect the results
        for future in as_completed(futures):
            try:
                base_filename, status_info = future.result()
                file_status_info[base_filename] = status_info
            except Exception as e:
                print(f"A critical error occurred in a worker process: {e}")

    # 6. Return tracker dictionary
    return file_status_info


# Divide trajectory files into training and testing sets
# Steps:
# 1. Create train and test folders
# 2. Identify all .traj files
# 3. Decide number of test files based on split ratio
# 4. Randomly select files for test
# 5. Move test and train files to respective folders
# 6. Count failed and duplicated files
# 7. Print summary
def split_dataset(output_dir: Path, split_ratio: float, seed: int, file_status_info: dict):
    print(f"\n--- Phase 2: Dividing files into training and testing sets (Test ratio: {split_ratio}) ---")
    
    # 1. Create train and test folders
    train_folder = output_dir / "train"
    test_folder = output_dir / "test"
    train_folder.mkdir(exist_ok=True)
    test_folder.mkdir(exist_ok=True)

    # 2. Identify all .traj files
    all_traj_files = list(output_dir.glob("*.traj"))
    if not all_traj_files:
        print("No trajectory files found to split.")
        return

    # 3. Decide number of test files based on split ratio
    random.seed(seed)
    test_count = int(split_ratio * len(all_traj_files))

    # 4. Randomly select files for test
    test_files = set(random.sample(all_traj_files, test_count))

    # 5. Move test and train files to respective folders
    for traj_file in all_traj_files:
        if traj_file in test_files:
            shutil.move(traj_file, test_folder / traj_file.name)
        else:
            shutil.move(traj_file, train_folder / traj_file.name)

    # 6. Count failed and duplicated files
    failed_count = sum(1 for v in file_status_info.values() if v["Status"] == "Failed")
    duplicated_count = sum(1 for v in file_status_info.values() if v["Duplicated"] == "Y")
    total_unused = failed_count + duplicated_count

    # 7. Print summary
    print(f"Total test files: {len(test_files)}")
    print(f"Total train files: {len(all_traj_files) - len(test_files)}")
    print(f"Total unused files: {total_unused} ({failed_count} failed, {duplicated_count} duplicated)")


## Generate CSV summary of all trajectory files
# Steps:
# 1. Print starting message
# 2. Define folders to check (train, test, unused)
# 3. Define CSV header
# 4. Open CSV file for writing
# 5. Loop over folders and trajectory files
#     5a. Prepare base filename and tracker info
#     5b. Try reading trajectory
#     5c. Extract last frame properties
#     5d. Determine lattice type, defects, surface, deformation, strain
#     5e. Extract chemical formula, species, number of atoms, volume
#     5f. Extract energy per atom, total energy
#     5g. Extract forces: max force
#     5h. Write row to CSV
# 6. Add rows for completely failed files
# 7. Print completion message
def generate_summary_csv(output_dir: Path, csv_path: Path, file_status_info: dict):
    # 1. Print starting message
    print(f"\n--- Phase 3: Creating summary CSV ({csv_path.name}) ---")

    # 2. Define folders to check (train, test, unused)
    folders_to_check = {
        "train": output_dir / "train",
        "test": output_dir / "test",
        "unused": output_dir / "unused_data"
    }

    # 3. Define CSV header
    header = [
        "Filename", "Dataset", "Duplicated", "Num_Frames",
        "Chemical_Formula", "Atomic_Species", "Lattice",
        "Defect_Type", "Surface", "Deform_Mode", "Strain_Value", "Num_Atoms",
        "Volume_A3", "Volume_per_Atom_A3", "Total_Energy_eV", "Energy_per_Atom_eV",
        "Max_Force_eV_A", "Notes"
    ]

    # 4. Open CSV file for writing
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        # 5. Loop over folders and trajectory files
        for dataset_name, folder_path in folders_to_check.items():
            if not folder_path.exists():
                continue

            for traj_file in folder_path.glob("*.traj"):
                # 5a. Prepare base filename and tracker info
                filename = traj_file.name
                base_key = filename.replace("_original.traj", "").replace(".traj", "")
                base_key = re.sub(r'_supercell_\d+x\d+x\d+', '', base_key)

                info = file_status_info.get(base_key, {})
                duplicated_status = info.get("Duplicated", "N")
                notes = info.get("Notes", "N/A")

                try:
                    # 5b. Try reading trajectory
                    frames = read(traj_file, index=":")
                    num_frames = len(frames)
                    last_frame = frames[-1]

                    # 5c. Extract last frame properties
                    lattice = "BCC" if "BCC" in filename else ("FCC" if "FCC" in filename else "Unknown")
                    is_surface = "surf" in filename
                    defect_match = re.search(r"(SIA\d*)", filename)
                    defect_type = defect_match.group(1) if defect_match else "None"
                    deform_match = re.search(r"(C11_C12_I{1,2}|C44)", filename)
                    deform_mode = deform_match.group(1) if deform_match else "None"
                    strain_match = re.search(r"strain_([+-]?\d*\.\d+)", filename)
                    strain_val = float(strain_match.group(1)) if strain_match else None

                    # 5d. Extract chemical formula, species, number of atoms, volume
                    formula = last_frame.get_chemical_formula()
                    species = str(list(set(last_frame.get_chemical_symbols())))
                    num_atoms = len(last_frame)
                    volume = last_frame.get_volume()
                    volume_per_atom = volume / num_atoms

                    # 5e. Extract energy per atom, total energy
                    try:
                        total_energy = last_frame.get_potential_energy()
                        energy_per_atom = total_energy / num_atoms
                    except Exception:
                        total_energy, energy_per_atom = None, None

                    # 5f. Extract forces: max force
                    try:
                        forces = last_frame.get_forces()
                        max_force = np.max(np.linalg.norm(forces, axis=1))
                    except Exception:
                        max_force = None

                    # 5g. Write row to CSV
                    row = [
                        filename, dataset_name, duplicated_status, num_frames, formula,
                        species, lattice, defect_type, is_surface, deform_mode,
                        strain_val, num_atoms, round(volume, 4), round(volume_per_atom, 4),
                        round(total_energy, 4) if total_energy else "None",
                        round(energy_per_atom, 4) if energy_per_atom else "None",
                        round(max_force, 4) if max_force else "None",
                        notes
                    ]
                    writer.writerow(row)

                except Exception as e:
                    print(f"Failed to write CSV row for {filename}: {e}")

        # 6. Add rows for completely failed files
        for base_key, info in file_status_info.items():
            if info["Status"] == "Failed":
                failed_name = info.get("Failed_File", f"{base_key}_OUTCAR")
                row = [failed_name, "unused", info["Duplicated"]] + ["None"] * (len(header)-4) + [info["Notes"]]
                writer.writerow(row)

    # 7. Print completion message
    print(f"Success! Summary saved to {csv_path}\n")


# Main function to parse arguments and run all steps
def main():
    parser = argparse.ArgumentParser(description="Extract VASP files, optionally enlarge boxes, split, and summarise DFT data.")
    parser.add_argument("--input_dir", type=Path, default=Path("data/Li-metal_OUTCARs"), help="VASP output directory.")
    parser.add_argument("--output_dir", type=Path, default=Path("data/traj"), help="Folder to store .traj files.")
    parser.add_argument("--csv_name", type=str, default="dft_summary.csv", help="Name of summary CSV file.")
    parser.add_argument("--split_ratio", type=float, default=0.10, help="Fraction of data for test set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument("--duplicate", type=int, choices=[1, 0], default=0, help="Enable enlargement of small boxes (1=on, 0=off).")
    parser.add_argument("--ffield", type=Path, default=Path("ffield.json"), help="Path to ffield.json to read rcut.")
    parser.add_argument("--cutoff", type=float, default=4.0, help="Fallback cutoff if ffield.json missing or unreadable.")
    
    # New Multiprocessing Argument
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Number of CPU cores to use for reading data (defaults to all cores).")
    
    args = parser.parse_args()

    actual_cutoff = get_cutoff_radius(args.ffield, args.cutoff)
    enlarge = args.duplicate == 1

    csv_path = args.output_dir / args.csv_name

    file_status_info = extract_and_process(args.input_dir, args.output_dir, actual_cutoff, enlarge, args.workers)
    split_dataset(args.output_dir, args.split_ratio, args.seed, file_status_info)
    generate_summary_csv(args.output_dir, csv_path, file_status_info)


if __name__ == "__main__":
    main()
