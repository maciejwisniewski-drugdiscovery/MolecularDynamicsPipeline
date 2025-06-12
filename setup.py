from setuptools import setup, find_packages

setup(
    name='plinder_dynamics',                  # Nazwa paczki
    version='0.1.0',                    # Wersja
    description='Molecular Dynamics Pipeline for PLINDER Protein - Ligand Complexes',  # Opis (opcjonalny)
    author='Maciej Wisniewski',
    author_email='m.wisniewski@datascience.edu.pl',
    packages=find_packages(where='src'),
    install_requires=[
        'numpy>=1.21.0',
        'openmm>=7.7.0',
        'openmmforcefields>=0.11.2',
        'openff-toolkit>=0.11.0',
        'pdbfixer>=1.8.0',
        'openbabel>=3.1.0',
        'rdkit>=2022.9.1',
        'pyyaml>=6.0.0',
        'pandas>=1.5.0',
        'tqdm>=4.65.0',
        'pathlib>=1.0.1'
    ],
    package_dir={'': 'src'},
    python_requires='>=3.7',
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)





platform = Platform.getPlatformByName('CUDA')
platform.setPropertyDefaultValue('Precision', 'mixed')
platform.setPropertyDefaultValue('CudaDeviceIndex', '0')


# bialka: PDB
# ligandzie: SDF


# .pdbqt