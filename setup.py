from setuptools import setup, find_packages


with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name='bcblib',        # This is the name of your PyPI-package.
    version='0.1.1',     # Update the version number for new releases
    # data_files=[('priors', ['../Data/ants_priors/brainPrior.nii.gz'])],
    keywords='brain neuroimaging',
    packages=find_packages(exclude=['__pycache__']),
    install_requires=['nibabel', 'numpy', 'six', 'os', 'sys', 'errno', 'scipy'],
    project_urls={  # Optional
        'Source': 'https://github.com/chrisfoulon/BCBlib',
        'Bug Reports': 'https://github.com/chrisfoulon/BCBlib/issues',
        'BCBlab website': 'http://bcblab.com'
    }
    )
