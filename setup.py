# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
#from setuptools.config import read_configuration

#conf_dict = read_configuration('./setup.cfg')

setup(
    name = "lerp",
    author =" Emmanuel Roux",
    version = '0.1aN',
    packages = find_packages(exclude=['build', 'contrib', 'docs', 'tests', 'sample']),
    install_requires = ['numpy', 'scipy', 'matplotlib', 'pandas'],
    license = "MIT",
    description = "Lookup table facility in python on top of numpy",
    long_description = read("README.rst"),
    keywords = "interpolation, lookup table",
    url = "https://github.com/gwin-zegal/lerp",
    download_url = "https://github.com/gwin-zegal/lerp/releases/tag/untagged-01068bebf35469123485",
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        ],

)
