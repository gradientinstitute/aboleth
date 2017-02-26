#!/usr/bin/env python

from setuptools import setup, find_packages

readme = open('README.md').read()

setup(
    name='aboleth',
    version='0.1.0',
    description='Bayesian tensorflow tools',
    long_description=readme,
    author='Determinant',
    author_email='lachlan.mccalman@data61.csiro.au',
    url='https://github.com/determinant-io/aboleth',
    packages=find_packages(),
    package_dir={'aboleth': 'aboleth'},
    include_package_data=True,
    install_requires=[
        'six==1.10.0',
        'PyContracts==1.7.15',
        'click==6.7',
        'bokeh==0.12.4',
        'numpy==1.12.0',
        'scipy==0.18.1',
        'scikit-learn==0.18.1',
        # 'tensorflow-gpu==1.0.0',
    ],
    extras_require={
        'dev': [
            'sphinx==1.4.8',
            'sphinxcontrib-programoutput==0.8',
            'pytest==3.0.3',
            'pytest-cov==2.4.0',
            'pytest-regtest==0.15.0',
            'flake8==3.0.4',
            'flake8-docstrings==1.0.2',
            'pydocstyle==1.1.1',
            'pyflakes==1.2.3',
            'mccabe==0.5.2',
            'pytest-flake8==0.8.1'
        ]
    },
    license="All Right Reserved",
    zip_safe=False,
    keywords='aboleth',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        "Operating System :: POSIX",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.4",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
)
