import subprocess
import sys
import os
from pathlib import Path
import argparse
from nilearn import datasets
from nilearn.regions import connected_regions

from bcblib.tools.general_utils import open_json
from bcblib.tools.nifti_utils import load_nifti


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


def test_parcitron(input_arg, input_value, output_path):
    # Determine the directory where the current script is located
    script_dir = Path(__file__).parent

    # Path to the file containing the voxel paths
    yeo_7_networks_voxels_path = script_dir / "yeo_7_networks_voxels.txt"
    yeo_17_networks_voxels_path = script_dir / "yeo_17_networks_voxels.txt"

    commands = [
        # KMeans with Parcel Size List
        f'parcitron -{input_arg} "{input_value}" -o "{output_path}/KMeans_parcel_sizes" --method KMeans '
        f'-rsl 150000,62000 --random_state 42',

        # KMeans with Number of Parcels
        f'parcitron -{input_arg} "{input_value}" -o "{output_path}/KMeans_num_parcels" --method KMeans -np 7,17 '
        f'--random_state 42',

        # Compactor with Fixed Size (Non-Contiguous)
        f'parcitron -{input_arg} "{input_value}" -o "{output_path}/Compactor_fixed_noncontig" --method compactor '
        f'-rsl 150000,62000 --strategy fixed_size --random_state 42',

        # Compactor with Fixed Size (Contiguous)
        f'parcitron -{input_arg} "{input_value}" -o "{output_path}/Compactor_fixed_contig" --method compactor '
        f'-rsl 150000,62000 --strategy fixed_size --contiguous --random_state 42',

        # Compactor with Balanced Size (Non-Contiguous)
        f'parcitron -{input_arg} "{input_value}" -o "{output_path}/Compactor_balanced_noncontig" --method compactor '
        f'-rsl 150000,62000 --strategy balanced_size --random_state 42',

        # Compactor with Balanced Size (Contiguous)
        f'parcitron -{input_arg} "{input_value}" -o "{output_path}/Compactor_balanced_contig" --method compactor '
        f'-rsl 150000,62000 --strategy balanced_size --contiguous --random_state 42',

        # Compactor with Number of Parcels (Non-Contiguous)
        f'parcitron -{input_arg} "{input_value}" -o "{output_path}/Compactor_num_parcels_noncontig" --method compactor '
        f'-np 7,17 --random_state 42',

        # Compactor with Number of Parcels (Contiguous)
        f'parcitron -{input_arg} "{input_value}" -o "{output_path}/Compactor_num_parcels_contig" --method compactor '
        f'-np 7,17 --contiguous --random_state 42',

        # Compactor with Custom Sizes (Contiguous) 7 Networks
        f'parcitron -{input_arg} "{input_value}" -o "{output_path}/Compactor_custom_sizes_contig_7" --method compactor '
        f'-cs {yeo_7_networks_voxels_path} --contiguous --random_state 42',

        # Compactor with Custom Sizes (Contiguous) 17 Networks
        f'parcitron -{input_arg} "{input_value}" -o "{output_path}/Compactor_custom_sizes_contig_17" '
        f'--method compactor -cs {yeo_17_networks_voxels_path} --contiguous --random_state 42',

        # Compactor with Custom Sizes (Non-Contiguous) 7 Networks
        f'parcitron -{input_arg} "{input_value}" -o "{output_path}/Compactor_custom_sizes_noncontig_7" '
        f'--method compactor -cs {yeo_7_networks_voxels_path} --random_state 42',

        # Compactor with Custom Sizes (Non-Contiguous) 17 Networks
        f'parcitron -{input_arg} "{input_value}" -o "{output_path}/Compactor_custom_sizes_noncontig_17" '
        f'--method compactor -cs {yeo_17_networks_voxels_path} --random_state 42',
    ]

    # Run each command and check the corresponding output directory
    for command in commands:
        print(f"Running: {command}")
        if run_command(command):
            # Extract the output directory from the command
            output_dir = command.split('-o ')[1].split(' ')[0].strip('"')
            check_output_dir(output_dir)


def run_custom_command_with_default(output_path):
    """Run the parcitron command with the default Yeo atlas as input and check for contiguity."""
    yeo_atlas = datasets.fetch_atlas_yeo_2011(data_dir=None, url=None, resume=True, verbose=1)
    yeo_17_nifti_path = yeo_atlas.thick_17
    input_arg = 'm'
    input_value = yeo_17_nifti_path

    # Path to the file containing the voxel paths
    script_dir = Path(__file__).parent
    yeo_17_networks_voxels_path = script_dir / "yeo_17_networks_voxels.txt"

    command = (
        f'parcitron -{input_arg} "{input_value}" -o "{output_path}/Compactor_custom_sizes_test_contig_17" '
        f'--method compactor -cs {yeo_17_networks_voxels_path} --contiguous --random_state 42'
    )

    print(f"Running: {command}")
    if run_command(command):
        output_dir = Path(output_path) / "Compactor_custom_sizes_test_contig_17" / "custom_sizes"
        json_path = output_dir / "__parcels_dict.json"

        assert json_path.exists(), f"JSON file {json_path} does not exist."
        print(f"Loading {json_path}")
        parcels_dict = open_json(json_path)

        contiguous_check = True
        for key in list(parcels_dict.keys())[:-1]:  # Skip the last key
            for path in parcels_dict[key]:
                nii_file = Path(path)
                assert nii_file.exists(), f"NIfTI file {nii_file} does not exist."

                print(f"Checking contiguity for {nii_file}")
                cluster_img, index = connected_regions(load_nifti(nii_file), min_region_size=1,
                                                       extract_type='connected_components', smoothing_fwhm=0)

                if cluster_img.shape[-1] == 1:
                    print(f"Parcel {key}: 1 contiguous cluster.")
                else:
                    contiguous_check = False
                    print(f"Parcel {key}: More than 1 cluster.")

        if contiguous_check:
            print("All parcels have exactly 1 contiguous cluster.")
        else:
            print("Not all parcels are contiguous.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the parcitron tool with various configurations.")
    parser.add_argument("input_arg", type=str,
                        help="Argument type: p, li, or m. (parcitron arg without the -)")
    parser.add_argument("input_value", type=str, help="Input value corresponding to the input argument.")
    parser.add_argument("output_path", type=str, help="Output path for the parcitron command results.")
    parser.add_argument("--default", action="store_true", help="Use default Yeo atlas as input for custom command.")

    print("Arguments:", sys.argv)
    args = parser.parse_args()

    if args.default:
        run_custom_command_with_default(args.output_path)
    else:
        if args.input_arg == 'default':
            yeo_atlas = datasets.fetch_atlas_yeo_2011(data_dir=None, url=None, resume=True, verbose=1)
            yeo_17_nifti_path = yeo_atlas.thick_17
            args.input_arg = 'm'
            args.input_value = yeo_17_nifti_path

        test_parcitron(args.input_arg, args.input_value, args.output_path)
