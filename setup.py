#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='deepnets',
    version='0.1.0',
    description='Bayesian Deep Nets',
    author='Determinant',
    author_email='determinant@data61.csiro.au',
    packages=["deepnets"],
    install_requires=[
        'numpy>=1.12.0',
        'scipy>=0.18.1',
        'tensorflow-gpu>=1.0.0',
        'scikit-learn>=0.18.1'
    ]
)
