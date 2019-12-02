#!/usr/bin/env python

import os

from setuptools import setup, find_packages

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='simba',
    version='0.1.1',
    packages=find_packages('.', exclude=('tests',)),
    zip_safe=True,
    include_package_data=False,
    description='Semantic similarity measures from Babylon Health',
    author='Babylon Health',
    license='Proprietary',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/babylonhealth/simba',
    install_requires=install_requires,
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Internet :: WWW/HTTP'
    ],
    entry_points={
        'console_scripts': [
            'simba = simba.__main__:simba',
        ]
    },
)
