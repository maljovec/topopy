# topopy
[![Build Status](https://travis-ci.org/maljovec/topopy.svg?branch=master)](https://travis-ci.org/maljovec/topopy)
[![Coverage Status](https://coveralls.io/repos/github/maljovec/topopy/badge.svg?branch=master)](https://coveralls.io/github/maljovec/topopy?branch=master)

A library for computing topological data structures stemming from Morse Theory. Given a set of arbitrarily arranged points in any dimension, this library is able to construct approximate topological structures using a neighborhood graph to simulate manifold structures.


# Example Usage

```python
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
```