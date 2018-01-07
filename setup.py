# -*- coding: utf-8 -*-

import os
import numpy as np
from Cython.Distutils import build_ext
from setuptools import setup, find_packages, Extension
# from setuptools.config import read_configuration
# conf_dict = read_configuration('./setup.cfg')


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

ext_modules = [ Extension('lerp.core.libNDTable',
                          sources = ['lerp/C/src/Python.c',
                                     'lerp/C/src/Core.c',
                                     'lerp/C/src/Interpolation.c'],
                          include_dirs = [np.get_include(), 'lerp/C/include']
                          ),
                # Extension("lerp.mesh",
                #         ["lerp/mesh.pyx"],
                #         language='c',)
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
    description="Lookup table facility in python on top of numpy", download_url="https://github.com/gwin-zegal/lerp/releases/\
        tag/untagged-01068bebf35469123485",
#    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas'],
    keywords="interpolation, lookup table",
    license="MIT",
    long_description=read("README.rst"),
    name="lerp",
    packages=find_packages(exclude=['build', 'contrib', 'docs', 'tests',
                                    'sample']),
    url="https://github.com/gwin-zegal/lerp",
    version='0.1',
)
