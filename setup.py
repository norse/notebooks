import setuptools
from setuptools import setup

import os
from os import path

pwd = path.abspath(path.dirname(__file__))

with open(path.join(pwd, "requirements.txt")) as fp:
    install_requires = fp.read()

setup(
    install_requires=install_requires,
    name="norse-notebooks",
    version="0.0.1",
    description="Tutorial notebooks for Norse - a library for deep learning with spiking neural networks",
    url="http://github.com/norse/notebooks",
    author="Christian Pehle, Jens E. Pedersen",
    author_email="christian.pehle@gmail.com, jens@jepedersen.dk",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="machine learning spiking neural networks",
)
