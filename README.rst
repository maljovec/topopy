topopy
======

.. badges

.. image:: https://img.shields.io/pypi/v/topopy.svg
        :target: https://pypi.python.org/pypi/topopy
        :alt: PyPI

.. image:: https://travis-ci.org/maljovec/topopy.svg?branch=master
        :target: https://travis-ci.org/maljovec/topopy
        :alt: Build Status

.. image:: https://coveralls.io/repos/github/maljovec/topopy/badge.svg?branch=master
        :target: https://coveralls.io/github/maljovec/topopy?branch=master
        :alt: Coverage Status

.. image:: https://readthedocs.org/projects/topopy/badge/?version=latest
        :target: https://topopy.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/maljovec/topopy/shield.svg
        :target: https://pyup.io/repos/github/maljovec/topopy/
        :alt: Pyup

.. end_badges

.. logo

.. image:: docs/_static/topopy.png
    :align: center
    :alt: topopy

.. end_logo

.. introduction

A library for computing topological data structures stemming from Morse Theory. Given a set of arbitrarily arranged points in any dimension, this library is able to construct approximate topological structures using a neighborhood graph to simulate manifold structures.

.. end_introduction

.. installation

Installation
~~~~~~~~~~~~

Currently, you can use pip_ to install this package
and all of its prerequisite libraries::

    pip install topopy

.. _pip: https://pip.pypa.io/en/stable/

Or to install from source, install all of the prerequiste libraries:

* scipy_
* numpy_
* scikit-learn_
* networkx_
* nglpy_

.. _scipy: https://www.scipy.org/
.. _numpy: http://www.numpy.org/
.. _scikit-learn: http://scikit-learn.org/)
.. _networkx: https://networkx.github.io/
.. _nglpy: https://github.com/maljovec/nglpy


And then clone and build the source repository::

    git clone https://github.com/maljovec/topopy.git
    cd topopy
    make
    python setup.py [develop|install]

.. end_installation

.. usage

Example Usage
~~~~~~~~~~~~~

::

    import topopy
    import numpy as np

    def hill(_x):
        _x = np.atleast_2d(_x)
        x = _x[:, 0]
        y = _x[:, 1]
        return np.exp(- ((x - .55)**2 + (y-.75)**2)/.125) + 0.01*(x+y)

    X = np.random.rand(100,2)
    Y = hill(X)

    msc = topopy.MorseSmaleComplex(graph='beta skeleton',
                                   gradient='steepest',
                                   normalization='feature',
                                   connect=True)
    msc.build(X, Y)
    msc.get_partitions()

.. end_usage
