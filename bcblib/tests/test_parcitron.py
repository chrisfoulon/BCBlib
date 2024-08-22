import subprocess
import sys
import os
from pathlib import Path
import argparse

def run_command(command):
    """Run a command in the shell."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed with error:\n{result.stderr}")
    else:
        print(result.stdout)
    return result.returncode == 0

def check_output_dir(output_dir):
    """Check if the output directory exists."""
    output_path = Path(output_dir)
    assert output_path.exists(), f"Output directory {output_dir} does not exist."
    print(f"Output directory {output_dir} exists.")

def test_parcitron(path, output_path):
    commands = [
        # KMeans with Parcel Size List
        f'parcitron -p "{path}" -o "{output_path}/KMeans_parcel_sizes" --method KMeans -rsl 30000,50000 --random_state 42',

        # KMeans with Number of Parcels
        f'parcitron -p "{path}" -o "{output_path}/KMeans_num_parcels" --method KMeans -np 50 --random_state 42',

        # Compactor with Fixed Size (Non-Contiguous)
        f'parcitron -p "{path}" -o "{output_path}/Compactor_fixed_noncontig" --method compactor -rsl 30000 --strategy fixed_size --random_state 42',

        # Compactor with Fixed Size (Contiguous)
        f'parcitron -p "{path}" -o "{output_path}/Compactor_fixed_contig" --method compactor -rsl 30000 --strategy fixed_size --contiguous --random_state 42',

        # Compactor with Balanced Size (Non-Contiguous)
        f'parcitron -p "{path}" -o "{output_path}/Compactor_balanced_noncontig" --method compactor -rsl 30000 --strategy balanced_size --random_state 42',

        # Compactor with Balanced Size (Contiguous)
        f'parcitron -p "{path}" -o "{output_path}/Compactor_balanced_contig" --method compactor -rsl 30000 --strategy balanced_size --contiguous --random_state 42',

        # Compactor with Number of Parcels (Non-Contiguous)
        f'parcitron -p "{path}" -o "{output_path}/Compactor_num_parcels_noncontig" --method compactor -np 20,30,50,100 --random_state 42',

        # Compactor with Number of Parcels (Contiguous)
        f'parcitron -p "{path}" -o "{output_path}/Compactor_num_parcels_contig" --method compactor -np 50 --contiguous --random_state 42',

        # Compactor with Custom Sizes (Contiguous)
        f'parcitron -p "{path}" -o "{output_path}/Compactor_custom_sizes_contig" --method compactor -cs 2000,4000,5000,2500 --contiguous --random_state 42',

        # Compactor with Custom Sizes (Non-Contiguous)
        f'parcitron -p "{path}" -o "{output_path}/Compactor_custom_sizes_noncontig" --method compactor -cs 2000,4000,5000,2500 --random_state 42',
    ]

    # Run each command and check the corresponding output directory
    for command in commands:
        print(f"Running: {command}")
        if run_command(command):
            # Extract the output directory from the command
            output_dir = command.split('-o ')[1].split(' ')[0].strip('"')
            check_output_dir(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the parcitron tool with various configurations.")
    parser.add_argument("path", type=str, help="Input path for the parcitron command.")
    parser.add_argument("output_path", type=str, help="Output path for the parcitron command results.")

    args = parser.parse_args()
    test_parcitron(args.path, args.output_path)
