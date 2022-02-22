from setuptools import setup

setup(
    name='SafeRLBench',
    version='1.0.1',
    author='Nicolas Ochsner',
    author_email='ochsnern@student.ethz.ch',
    packages=[
        'SafeRLBench',
        'SafeRLBench.algo',
        'SafeRLBench.envs',
        'SafeRLBench.spaces',
        'SafeRLBench.policy',
    ],
    description='Safe Reinforcement Learning Benchmark',
    keywords='reinforcement-learning benchmark',
    url='https://github.com/befelix/Safe-RL-Benchmark',
    install_requires=[
        'numpy >= 1.7',
        'scipy >= 0.19.0',
        'six >= 1.10',
        'futures >= 3.0.5;python_version<"3.2"'
    ],
    extras_require={
        'gym': ['gym >= 0.8.0'],
        'safeopt': ['GPy >= 1.6.1', 'safeopt >= 0.1'],
        'neural': ['tensorflow >= 1.0.0'],
    },
    dependency_links=[
        'git+https://github.com/befelix/SafeOpt/tarball/master#egg=safeopt-0.1'
    ],
)
