from bcblib.tools.growth import evolve_blobs
from bcblib.tools.blob_robot_core import initialize_blobs_with_mask, cell_array_to_state_array
from bcblib.tools.shapes import create_spherical_shape
from nilearn import datasets
from nilearn.plotting import plot_stat_map
import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time

# Fix the random seed for reproducibility
np.random.seed(42)

# Measure time for fetching the Yeo atlas
start_time = time()
# Fetch the Yeo 2011 atlas
yeo_atlas = datasets.fetch_atlas_yeo_2011()
fetch_time = time() - start_time
print(f"Time taken to fetch Yeo atlas: {fetch_time:.2f} seconds")

# Load the 17-network parcellation NIfTI file
yeo_17_nifti_path = yeo_atlas.thick_17
print("17-Network Yeo parcellation NIfTI file path:", yeo_17_nifti_path)

# Load the Yeo parcellation as a NIfTI image
yeo_img = nib.load(yeo_17_nifti_path)
print("Yeo parcellation NIfTI image shape:", yeo_img.shape)

# Remove the last dimension if it is 1 and recreate the Nifti1Image
if len(yeo_img.shape) == 4 and yeo_img.shape[-1] == 1:
    yeo_img = nib.Nifti1Image(yeo_img.get_fdata()[:, :, :, 0], yeo_img.affine)

# Convert the Yeo parcellation to a binary mask
yeo_data = yeo_img.get_fdata()
binary_mask = (yeo_data > 0).astype(int)

# Use the binary mask's header for the template
template_hdr = yeo_img

# Find non-zero regions in the mask
non_zero_indices = np.argwhere(binary_mask)

# Randomly choose 3 seed coordinates from the non-zero regions
chosen_indices = non_zero_indices[np.random.choice(non_zero_indices.shape[0], 3, replace=False)]
seed_coords = [tuple(idx) for idx in chosen_indices]

# Create a spherical shape with a radius of 3
spherical_shape = create_spherical_shape(radius=3)

# Print the chosen seed coordinates for reference
print("Chosen seed coordinates:", seed_coords)

# Define seed values (cluster indices)
seed_values = [1, 2, 3]

# Measure time for initialization of the cell array and blobs
start_time = time()
# Initialize the blobs and cell array with the mask and seed values
blobs, cell_array = initialize_blobs_with_mask(binary_mask.shape, binary_mask, seed_coords, seed_values,
                                               max_size=20000, seed_shape=spherical_shape)  # Set max_size to 20000
init_time = time() - start_time
print(f"Time taken to initialize blobs and cell array: {init_time:.2f} seconds")

# Visualize the initial state
start_time = time()
data = cell_array_to_state_array(cell_array, value_mode='state')
new_img = nib.Nifti1Image(data + 1, template_hdr.affine, dtype=np.uint16)
centre_of_mass = ndimage.center_of_mass(data + 1)
print(f'Number of live cells: {data[np.where(data > 0)].size}')
print(f'Coordinates of the centre of mass: {centre_of_mass}')

# Manually setting the cut coordinates in case the centre_of_mass has less or more than 3 values
if len(centre_of_mass) != 3:
    centre_of_mass = (int(data.shape[0] / 2), int(data.shape[1] / 2), int(data.shape[2] / 2))

# Visualization using plot_stat_map
plot_stat_map(new_img, output_file=None, cut_coords=seed_coords[0], cmap='hsv', colorbar=True, draw_cross=False)
plt.show()
visualization_time = time() - start_time
print(f"Time taken for initial visualization: {visualization_time:.2f} seconds")

# Evolve the blobs (parcellation)
for it in tqdm(range(100)):
    start_time = time()
    evolve_blobs(blobs, cell_array, it_time=it)
    evolve_time = time() - start_time
    print(f"Time taken for iteration {it}: {evolve_time:.2f} seconds")

    # Check if all blobs are complete and stop if so
    if all(blob.growth_complete for blob in blobs):
        print("All blobs have completed growth.")
        break

    if not any(blob.can_grow() for blob in blobs):
        print("None of the blobs can grow further.")
        break

    # Visualize intermediate steps (optional, e.g., every 10 iterations)
    # if it % 10 == 0:
    #     start_time = time()
    #     data = cell_array_to_state_array(cell_array, value_mode='state')
    #     print(f'Number of live cells: {data[np.where(data > 0)].size}')
    #     new_img = nib.Nifti1Image(data + 1, template_hdr.affine, dtype=np.uint16)
    #     plot_stat_map(new_img, output_file=None, cut_coords=seed_coords[0], cmap='hsv', colorbar=True, draw_cross=False)
    #     plt.show()
    #     visualization_time = time() - start_time
    #     print(f"Time taken for visualization at iteration {it}: {visualization_time:.2f} seconds")

# Final visualization and saving the result
start_time = time()
data = cell_array_to_state_array(cell_array, value_mode='state')
print(f'Number of live cells: {data[np.where(data > 0)].size}')
print(f'Number of unique values: {np.unique(data)}')

# Print size of each blob and number of voxels for each label
for blob in blobs:
    print(f'Blob size: {blob.size}')

unique_labels, counts = np.unique(data, return_counts=True)
label_counts = dict(zip(unique_labels, counts))
print("Number of voxels per label:", label_counts)

# Save the final NIfTI image
new_img = nib.Nifti1Image(data + 1, template_hdr.affine, dtype=np.uint16)
nib.save(new_img, r'C:\Users\Tolhsadum\python_data\test_blob_robot_parcellation_blobs.nii.gz')
plot_stat_map(new_img, output_file=None, cut_coords=seed_coords[0], cmap='hsv', colorbar=True, draw_cross=False)
plt.show()
final_save_time = time() - start_time
print(f"Time taken for final visualization and saving: {final_save_time:.2f} seconds")
