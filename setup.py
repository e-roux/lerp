# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from setuptools.config import read_configuration

conf_dict = read_configuration('./setup.cfg')

setup(
    packages=find_packages(exclude=['build', 'contrib', 'docs', 'tests', 'sample']),
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas'],

)
