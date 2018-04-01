# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from setuptools import setup, find_packages, Extension
from setuptools.command.test import test as TestCommand

# from setuptools.config import read_configuration
# conf_dict = read_configuration('./setup.cfg')


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# http://setuptools.readthedocs.io/en/latest/setuptools.html#test-build-package-and-run-a-unittest-suite
# https://docs.pytest.org/en/latest/goodpractices.html#manual-integration
class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


extra_compile_args = ['-Wall', '-Wno-unused-function', '-Wno-unused-variable']\
    if os.name == 'posix' else ['-Wall']

ext_modules = [Extension('lerp.core.interpolation',
                         sources=['lerp/C/src/LERP_intern.c',
                                  'lerp/C/src/NDTable.c',
                                  'lerp/C/src/Mesh.c',
                                  'lerp/C/src/interpolation.c'],
                         include_dirs=[np.get_include(),
                                       'lerp/C/include'],
#                         libraries=["gsl"],
                         extra_compile_args=extra_compile_args),
               Extension('lerp.core.utils',
                         sources=['lerp/C/src/LERP_intern.c',
                                  'lerp/C/src/utils.c',],
                         include_dirs=[np.get_include(),
                                       'lerp/C/include'],
                         extra_compile_args=extra_compile_args),
               ]

setup(
    author=" Emmanuel Roux",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
    description="Lookup table facility in python on top of numpy",
    download_url="https://github.com/gwin-zegal/lerp/releases/\
        tag/untagged-01068bebf35469123485",
    ext_modules=ext_modules,
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas', 'xarray'],
    keywords="interpolation, lookup table",
    license="MIT",
    long_description=read("README.rst"),
    name="lerp",
    packages=find_packages(exclude=['benchmark', 'build', 'contrib',
                                    'docs', 'tests', 'sample']),
    tests_require=['pytest'],
    cmdclass={'test': PyTest},
    url="https://github.com/gwin-zegal/lerp",
    version='0.1',
)
