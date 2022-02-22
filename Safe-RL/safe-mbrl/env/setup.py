#!/usr/bin/env python

from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "Safety Gym is designed to work with Python 3.6 and greater. " \
    + "Please install it before proceeding."

setup(
    name='safety_gym',
    packages=['safety_gym'],
    install_requires=[
        'gym~=0.15.3',
        'joblib~=0.14.0',
        'mujoco_py==2.0.2.7',
        'numpy~=1.17.4',
        'xmltodict~=0.12.0',
    ],
)
