#! /usr/bin/env python
from setuptools import setup, find_packages

# Detect minimum version of python
import sys
if sys.version_info < (3, 5):
    sys.exit('Aboleth requires Python 3.5+')

exec(open('aboleth/version.py').read())
readme = open('README.rst').read()

setup(
    name='aboleth',
    version=__version__,
    description='Bayesian supervised deep learning with TensorFlow',
    long_description=readme,
    author='Daniel Steinberg, Lachlan McCalman',
    author_email='daniel.steinberg@data61.csiro.au',
    url='https://github.com/data61/aboleth',
    packages=find_packages(),
    package_dir={'aboleth': 'aboleth'},
    include_package_data=True,
    install_requires=[
        'numpy>=1.12.0',
        'scipy>=0.18.1',
        'tensorflow>=1.4.0',
        'six>=1.10.0'
    ],
    extras_require={
        'dev': [
            'sphinx>=1.4.8',
            'pytest>=3.1.0',
            'pytest-mock>=1.6.0',
            'pytest-cov>=2.5.1',
            'pytest-flake8>=0.8.1',
            'flake8-docstrings>=1.1.0',
        ],
        'demos': [
            'bokeh>=0.12.4',
            'pandas>=0.20.3',
            'scikit-learn>=0.18.1',
        ]
    },
    license="Apache 2.0",
    zip_safe=False,
    keywords='aboleth',
    classifiers=[
        'Development Status :: 4 - Beta',
        "Operating System :: POSIX",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
)
