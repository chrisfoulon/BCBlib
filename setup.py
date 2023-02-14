from setuptools import setup, find_packages
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='bcblib',        # This is the name of your PyPI-package.
    version='0.3.2.1',     # Update the version number for new releases
    # data_files=[('priors', ['../Data/ants_priors/brainPrior.nii.gz'])],
    keywords='brain neuroimaging nifti cellular-automata bcbtoolkit bcblab',
    long_description=read('README.rst'),
    zip_safe=True,
    include_package_data=True,
    packages=find_packages(exclude=['__pycache__']),
    install_requires=['nibabel>=3', 'numpy', 'six', 'scipy', 'nilearn', 'sklearn', 'tqdm', 'pandas', 'openpyxl'],
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
        'console_scripts': ['generate_synth_lesions = bcblib.scripts.generate_synth_lesions:main',
                            'pick_up_matched_synth_lesions = bcblib.scripts.pick_up_synth_lesions:main',
                            'randomise_helper = bcblib.tools.randomise_helper:randomise_helper',
                            'anacom2 = bcblib.anacom2.anacom2:anacom2']
        # 'console_scripts': ['dicom_conversion = data_identification.scripts.dicom_conversion:convert']
    },
    project_urls={  # Optional
        'Source': 'https://github.com/chrisfoulon/BCBlib',
        'Bug Reports': 'https://github.com/chrisfoulon/BCBlib/issues',
        'BCBlab website': 'http://bcblab.com'
    }
    )
