"""
      Setup script for topopy
"""
from setuptools import setup, Extension
import re
import platform

requirements = open('requirements.txt').read().strip().split('\n')

extra_compile_args = ["-O3", "-march=native", ]
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args.append('-stdlib=libc++')
    extra_link_args.append('-stdlib=libc++')


def get_property(prop, project):
    """
        Helper function for retrieving properties from a project's
        __init__.py file
        @In, prop, string representing the property to be retrieved
        @In, project, string representing the project from which we will
        retrieve the property
        @Out, string, the value of the found property
    """
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
        open(project + "/__init__.py").read(),
    )
    return result.group(1)


FILES = [
    "UnionFind.cpp",
    "MergeTree.cpp",
    "MorseComplex.cpp",
    "utils.cpp",
    "topology_wrap.cpp",
]
VERSION = get_property("__version__", "topopy")

# Consult here: https://packaging.python.org/tutorials/distributing-packages/
setup(
    name="topopy",
    packages=["topopy"],
    version=VERSION,
    description="A library for computing topological data structures",
    long_description="Given a set of arbitrarily arranged points in any "
    + "dimension, this library is able to construct "
    + "approximate topological structures using a "
    + "neighborhood graph.",
    author="Dan Maljovec",
    author_email="maljovec002@gmail.com",
    license="BSD",
    test_suite="topopy.tests",
    url="https://github.com/maljovec/topopy",
    download_url="https://github.com/maljovec/topopy/archive/"
    + VERSION
    + ".tar.gz",
    keywords=[
        "topological data analysis",
        "computational topology",
        "Morse theory",
        "merge tree",
        "contour tree",
        "extremum graph",
        "Morse-Smale complex",
        "Morse complex",
    ],
    # Consult here: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    setup_requires=["scipy", "numpy"],
    install_requires=requirements,
    python_requires=">=2.7, <4",
    ext_modules=[
        Extension("_topology",
                  FILES,
                  extra_compile_args=extra_compile_args,
                  extra_link_args=extra_link_args)
    ],
)
