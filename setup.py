from setuptools import setup, find_packages

setup(
    name='dynamics_pipeline',
    version='0.1.0',
    description='Molecular Dynamics Pipeline for Biomolecule Complexes',
    author='Maciej Wisniewski',
    author_email='m.wisniewski@datascience.edu.pl',
    packages=find_packages(where='src'),
    install_requires=[
        'numpy>=1.21.0',
        'openmm>=7.7.0',
        'openmmforcefields @git+https://github.com/openmm/openmmforcefields.git',
        'pint>=0.20.1',
        'openff-units @git+https://github.com/openforcefield/openff-units.git',
        'openff-utilities @git+https://github.com/openforcefield/openff-utilities.git',
        'openff-toolkit @git+https://github.com/openforcefield/openff-toolkit.git@0.16.9',
        'pdbfixer @git+https://github.com/openmm/pdbfixer.git',
        'pdb2pqr>=3.0.0',
        'rdkit>=2022.9.1',
        'pyyaml>=6.0.0',
        'pandas>=1.5.0',
        'tqdm>=4.65.0',
        'pathlib>=1.0.1',
        'openbabel-wheel',
        'scipy>=1.10.0',
        'biotite==1.2.0',
        'cachetools',
        'plinder',
        'importlib-resources'
        'python-dotenv',
        'jinja2'
    ],
    package_dir={'': 'src'},
    python_requires='>=3.7, <3.12',
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)