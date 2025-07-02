from setuptools import setup, find_packages

setup(
    name='molecular_dynamics_pipeline',
    version='0.1.0',
    description='Molecular Dynamics Pipeline for Biomolecule Complexes',
    author='Maciej Wisniewski',
    author_email='m.wisniewski@datascience.edu.pl',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.7, <3.13',
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)