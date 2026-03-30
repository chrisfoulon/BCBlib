from setuptools import setup, find_packages
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='bcblib',        # This is the name of your PyPI-package.
    version='0.6.0',     # Update the version number for new releases
    # data_files=[('priors', ['../Data/ants_priors/brainPrior.nii.gz'])],
    keywords='brain neuroimaging nifti cellular-automata bcbtoolkit bcblab parcellation null-models',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    zip_safe=True,
    include_package_data=True,
    packages=find_packages(exclude=['__pycache__']),
    install_requires=['nibabel>=3', 'numpy', 'six', 'scipy', 'nilearn', 'scikit-learn',
                      'tqdm', 'pandas', 'openpyxl', 'umap-learn', 'joblib', 'statsmodels', 'mne',
                      'pymc>=5', 'arviz', 'matplotlib', 'rich>=10'],
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.rst"],
        # Include all the executables from bin
        "Data": ["*"],
        "bash": ["*"],
    },
    author='Chris Foulon, Michel Thiebaut de Schotten',
    author_email="hd.chrisfoulon@gmail.com",
    entry_points={
        'console_scripts': ['parcitron = bcblib.scripts.parcitron:main',
                            'pick_up_matched_synth_lesions = bcblib.scripts.pick_up_synth_lesions:main',
                            'randomise_helper = bcblib.tools.randomise_helper:randomise_helper',
                            'anacom2 = bcblib.anacom2.anacom2:anacom2',
                            'ml_results = bcblib.scripts.ml_results:main',
                            'bcb-info = bcblib.scripts.imaging_cli:bcb_info',
                            'bcb-header = bcblib.scripts.imaging_cli:bcb_header',
                            'bcb-stats = bcblib.scripts.imaging_cli:bcb_stats',
                            'bcb-orient = bcblib.scripts.imaging_cli:bcb_orient',
                            'bcb-roi = bcblib.scripts.imaging_cli:bcb_roi',
                            'bcb-merge = bcblib.scripts.imaging_cli:bcb_merge',
                            'bcb-split = bcblib.scripts.imaging_cli:bcb_split',
                            'bcb-convert = bcblib.scripts.imaging_cli:bcb_convert',
                            'bcb-dataset-split = bcblib.scripts.run_dataset_splitting:main']
    },
    project_urls={  # Optional
        'Source': 'https://github.com/chrisfoulon/BCBlib',
        'Bug Reports': 'https://github.com/chrisfoulon/BCBlib/issues',
        'BCBlab website': 'http://bcblab.com'
    }
    )
