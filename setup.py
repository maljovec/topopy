"""
      Setup script for topopy
"""
import platform

from setuptools import Extension, setup

extra_compile_args = [
    "-O3",
]
extra_link_args = []

if platform.system() == "Darwin":
    extra_compile_args.append("-stdlib=libc++")
    extra_link_args.append("-stdlib=libc++")


FILES = [
    "UnionFind.cpp",
    "MergeTree.cpp",
    "MorseComplex.cpp",
    "utils.cpp",
    "topology_wrap.cpp",
]

# Consult here: https://packaging.python.org/tutorials/distributing-packages/
setup(
    name="topopy",
    packages=["topopy"],
    description="A library for computing topological data structures",
    long_description="Given a set of arbitrarily arranged points in any "
    + "dimension, this library is able to construct "
    + "approximate topological structures using a "
    + "neighborhood graph.",
    test_suite="topopy.tests",
    python_requires=">=2.7, <4",
    ext_modules=[
        Extension(
            "topopy._topology",
            FILES,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
)
