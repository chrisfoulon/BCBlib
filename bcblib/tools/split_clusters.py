
import numpy as np

import nibabel as nib

def split_clusters(nii, res_folder, name):
    # if not os.path.exists(res_folder):
    #     os.mkdir(res_folder)
    # nii = nib.load(img)
    # affine = nii.affine
    data = nii.get_data()
    affine = nii.affine
    o_max = np.amax(data)

    # tt = []
    folder = os.path.join(res_folder, name)
    if not os.path.exists(folder):
        os.mkdir(folder)

    for i in np.arange(1, o_max + 1):
        clu = np.array(np.where(data == i))
        mask = np.zeros(data.shape)
        mask[clu[0,], clu[1,], clu[2,]] = i
        # tt.append(len(clu))
        img_ROIs = nib.Nifti1Image(mask, affine)
        path = os.path.join(folder, "clu" + str(i) + "_" + name + ".nii.gz")
        nib.save(img_ROIs, path)
