#!/usr/bin/env python

from setuptools import setup, find_packages

exec(open('aboleth/version.py').read())
readme = open('README.rst').read()

setup(
    name='aboleth',
    version=__version__,
    description='Bayesian supervised deep learning with tensorflow',
    long_description=readme,
    author='Determinant',
    author_email='lachlan.mccalman@data61.csiro.au',
    url='https://github.com/determinant-io/aboleth',
    packages=find_packages(),
    package_dir={'aboleth': 'aboleth'},
    include_package_data=True,
    install_requires=[
        'numpy>=1.12.0',
        'scipy>=0.18.1',
        'scikit-learn>=0.18.1',
        'six>=1.10.0',
        'bokeh>=0.12.4',
        # 'tensorflow-gpu>=1.2.0',
    ],
    extras_require={
        'dev': [
            'sphinx>=1.4.8',
            'pytest>=3.1.0',
            'pytest-cov>=2.5.1',
            'pytest-flake8>=0.8.1',
            'flake8-docstrings>=1.1.0',
        ]
    },
    license="All Right Reserved",
    zip_safe=False,
    keywords='aboleth',
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Operating System :: POSIX",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
)
