from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import sys
import pip

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

with open('requirements_dev.txt', 'r') as f:
    test_requirements = f.read().splitlines()

setup(
    name="safe_learning",
    version="0.0.1",
    author="Felix Berkenkamp",
    author_email="fberkenkamp@gmail.com",
    description=("An demonstration of how to create, document, and publish "
                  "to the cheese shop a5 pypi.org."),
    license="MIT",
    keywords="safe reinforcement learning Lyapunov",
    url="https://github.com/befelix/lyapunov-learning",
    packages=find_packages(exclude=['docs']),
    setup_requires=['numpy'],
    install_requires=requirements,
    extras_require={'test': list(test_requirements)},
    tests_require=test_requirements,
    dependency_links=['git+https://github.com/GPflow/GPflow.git@0.4.0#egg=gpflow-0.4.0'],
    cmdclass={'test': PyTest},
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
)
