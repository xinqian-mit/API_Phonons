#!python3
#cython: language_level=3

from Cython.Build import cythonize
from distutils.core import setup

setup(ext_modules = cythonize('AllanFeldman.pyx'),extra_link_args=[])

