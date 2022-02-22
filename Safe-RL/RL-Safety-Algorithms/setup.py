import setuptools
import sys

if sys.version_info.major != 3:
    raise TypeError(
        'This Python is only compatible with Python 3, but you are running '
        'Python {}. The installation will likely fail.'.format(
            sys.version_info.major))

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rl_safety_algorithms",  # this is the name displayed in 'pip list'
    version="0.1",
    author="Sven Gronauer",
    author_email="sven.gronauer@tum.de",
    description="Algorithms for Safe Reinforcement Learning Problems.",
    install_requires=[
        'mpi4py',  # can be skipped if you want to use single threads
        'numpy',
        'torch'
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sven.gronauer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
