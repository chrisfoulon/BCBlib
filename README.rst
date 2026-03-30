BCBlib
======

A Python library of neuroimaging utilities developed at the
`BCBlab <http://bcblab.com>`_.  It provides NIfTI image processing,
FSL workflow helpers, research-grade statistical tools, and a suite of
command-line tools for day-to-day MRI data analysis.

.. code-block:: bash

    pip install bcblib


Quick Start
-----------

.. code-block:: python

    # Load and inspect a NIfTI image
    from bcblib.imaging import load_nifti, image_stats
    nii = load_nifti("subject01_lesion.nii.gz")
    stats = image_stats(nii)

    # Balanced dataset splitting for cross-validation
    from bcblib.tools.dataset_splitting import permutation_balanced_splits
    folds, score, report = permutation_balanced_splits(
        groups=has_chronic,
        covariates={"acute_vol": acute_volumes, "chronic_vol": chronic_volumes},
        n_splits=5,
        n_permutations=50000,
        seed=42,
    )

    # Prepare FSL randomise inputs from a spreadsheet
    from bcblib.tools.randomise_helper import spreadsheet_to_mat_and_file_list
    spreadsheet_to_mat_and_file_list(
        "subjects.csv", columns=["age", "score"],
        output_dir="randomise_inputs/", filenames_column="image_path"
    )


Modules
-------

Imaging (``bcblib.imaging``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A modern FSL-equivalent API for NIfTI image processing.  All functions
accept either a file path or an already-loaded ``Nifti1Image``.

=========================  =====================================================
Module                     Description
=========================  =====================================================
``bcblib.imaging.io``      Load, save, resave NIfTI images; format detection
``bcblib.imaging.info``    Header inspection — equivalent to ``fslinfo``
``bcblib.imaging.stats``   Centre of gravity, volume, histogram, laterality
``bcblib.imaging.math``    Binarize, dilate, erode, apply/invert masks
``bcblib.imaging.orient``  Get and reorient image orientation
``bcblib.imaging.manipulate`` Extract ROI, merge, split 4-D images
``bcblib.imaging.convert`` Convert between ``.nii`` and ``.nii.gz``
=========================  =====================================================

.. note::
   ``bcblib.tools.nifti_utils``, ``bcblib.tools.images_utils``, and
   ``bcblib.tools.nii_stats`` are kept as backward-compatibility shims.
   New code should import from ``bcblib.imaging`` directly.

Research Tools (``bcblib.tools``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``best_overlap``
    Compute preserved structural connectivity from patient and cluster
    disconnectome maps using Bayesian modelling (PyMC).  Outputs latent
    connectivity scores with uncertainty estimates per brain region.
    Used in published research.

``dataset_splitting``
    Monte Carlo permutation search for the most balanced k-way dataset
    split.  Balances group counts (round-robin hard constraint) and
    continuous covariate distributions (Kruskal-Wallis minimax score).
    Returns fold indices, best score, and a full JSON report including
    convergence history and per-fold descriptive stats.
    Used in published research.

``randomise_helper``
    Build FSL ``randomise`` inputs from a spreadsheet: generates ``.mat``
    design files, concatenates 4-D NIfTI stacks, and manages file lists.
    Used in published research.

``split_clusters``
    Split a multi-label NIfTI atlas into one file per label value.

``divide_mask``
    Cluster a binary mask into spatially separate components by proximity.

``shapes``
    Generate geometric shapes (hyperspheres, arbitrary forms) as NIfTI
    arrays — useful for creating synthetic phantom data.

``constants``
    Pre-computed MNI 1 mm and 2 mm affines and shapes;
    ``empty_MNI1MM()`` / ``empty_MNI2MM()`` convenience constructors.

``general_utils``
    JSON I/O with support for NumPy arrays, UUIDs, and datetime objects.

``spreadsheet_io_utils``
    Load CSV and Excel files with column selection helpers.

``dataframe_filtering``
    Remove constant columns, filter by completeness threshold, handle
    datetime columns for ML pipelines.

``mat_transform``
    Connectivity matrix preprocessing: log\ :sub:`2` transform, z-score
    normalisation, rank transform.

``arrays_utils``
    Coordinate validation and centroid calculations for NumPy arrays.

``umap_utils``
    UMAP dimensionality reduction wrappers tuned for neuroimaging data.

``visualisation``
    Wrappers for MRIcron, matplotlib, and TensorBoard visualisation.

    .. note::
       Requires external tools (MRIcron, TensorBoard) installed separately
       depending on which functions are used.


CLI Tools
---------

Installed as console scripts:

===========================  ===================================================
Command                      Description
===========================  ===================================================
``bcb-info``                 Print NIfTI header info (``fslinfo`` equivalent)
``bcb-header``               Inspect and display full NIfTI header
``bcb-stats``                Image statistics (``fslstats`` equivalent)
``bcb-orient``               Get or set image orientation
``bcb-roi``                  Extract a region of interest
``bcb-merge``                Merge NIfTI images along a dimension
``bcb-split``                Split a 4-D NIfTI along the volume axis
``bcb-convert``              Convert between ``.nii`` and ``.nii.gz``
``bcb-dataset-split``        Balanced dataset splitting from a CSV file
``randomise_helper``         Build FSL randomise design files from a spreadsheet
``pick_up_matched_synth_lesions``  Select synthetic lesions matching a size distribution
===========================  ===================================================

Example — balanced split from the command line:

.. code-block:: bash

    bcb-dataset-split \
        --input subjects.csv \
        --group-col has_chronic \
        --covariate-cols acute_volume chronic_volume \
        --n-splits 5 --n-permutations 50000 --seed 42 \
        --output splits.csv
    # writes splits.csv and splits_report.json


Dependencies
------------

Core dependencies installed automatically:

- ``nibabel``, ``numpy``, ``scipy``, ``nilearn``, ``scikit-learn``
- ``pandas``, ``openpyxl``
- ``tqdm``, ``joblib``, ``statsmodels``
- ``matplotlib``, ``rich``
- ``pymc >= 5``, ``arviz`` (required for ``best_overlap``)
- ``umap-learn`` (required for ``umap_utils``)
- ``mne``

External tools (not installed by pip):

- FSL — required for ``randomise_helper`` to run ``randomise`` itself
- MRIcron — required for ``visualisation`` MRIcron wrappers
- TensorBoard — required for ``visualisation`` TensorBoard integration


Links
-----

- **Source**: https://github.com/chrisfoulon/BCBlib
- **Bug reports**: https://github.com/chrisfoulon/BCBlib/issues
- **BCBlab**: http://bcblab.com
- **PyPI**: https://pypi.org/project/bcblib/
