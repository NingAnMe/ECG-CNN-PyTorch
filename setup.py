#!/usr/bin/env python

import os
from setuptools import find_packages, setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()


setup(
    name='pytorch_utils',
    version='0.1.dev0',
    packages=find_packages(
        exclude=['tests', '*.tests', '*.tests.*', 'tests.*']
    ),
    package_dir={'pytorch_utils': os.path.join('.', 'pytorch_utils')},
    description='pytorch_utils - loss function, optimizer, lr scheduler',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    author='NingAnMe',
    license='Apache',
    install_requires=['torch']
)
