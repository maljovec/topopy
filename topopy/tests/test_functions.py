import random
import math
import numpy as np
import scipy
import copy


def generate_test_grid_2d(resolution=40):
    """
    """
    x, y = np.mgrid[0:1:(resolution * 1j), 0:1:(resolution * 1j)]
    return np.vstack([x.ravel(), y.ravel()]).T


def unpack2D(_x):
    """
        Helper function for splitting 2D data into x and y component to
        make equations simpler
    """
    _x = np.atleast_2d(_x)
    x = _x[:, 0]
    y = _x[:, 1]
    return x, y


###############################################################################
# Test Functions
###############################################################################

###############################################################################
# https://bit.ly/2wWSJs4


def torusPolyGeneratorF(x, n):
    product = 1
    for i in range(1, n + 1):
        product *= (x - (i - 1)) * (x - i)
    return product


def torusPolyGeneratorG(x, y, n):
    return torusPolyGeneratorF(x, n) + y ** 2


def doubleTorus(x, y, n=2, r=0.1, sgn=1):
    return sgn * np.sqrt(r ** 2 - torusPolyGeneratorG(x, y, n) ** 2)


# def torusPolyGeneratorF(x, y):
#     return (x**2+y**2)**2 - 4*x**2 + y**2


# def doubleTorus(x, y, r=0.1, sgn=1):
#     return sgn*np.sqrt(r**2 - torusPolyGeneratorF(x,y)**2)


def genTorusInputSampleSet(N):
    r = 0.175
    n = 2
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)

    # Solved the roots of f(x)=r^2 using wolfram alpha and truncate to
    # make sure we don't end up with negatives under the radical.
    # Note, changing r will affect these bounds
    minX = -0.073275
    maxX = n - minX

    for i in range(N):
        ySgn = (-1) ** random.randint(1, 2)
        zSgn = (-1) ** random.randint(1, 2)
        # x[i] = random.uniform(0, n)
        x[i] = random.uniform(minX, maxX)

        # The distribution below requires less samples, but may be
        # harder to sell in a paper, as it is clearly non-uniform,
        # whereas the sampling in y-space makes it look more uniform due
        # to the dependence on x which yields more samples closer to
        # zero, so we bias towards the more extreme values to counteract
        # this effect.

        # x[i] = random.betavariate(0.5,0.5)*(maxX-minX) + minX

        fx = torusPolyGeneratorF(x[i], n)
        minY = np.sqrt(max(-r - fx, 0))
        maxY = np.sqrt(r - fx)
        # y[i] = ySgn*(random.uniform(minY, maxY))
        y[i] = ySgn * (random.betavariate(0.5, 0.5) * (maxY - minY) + minY)
        z[i] = doubleTorus(x[i], y[i], n, r, zSgn)
    return (x, y, z)


# End https://bit.ly/2wWSJs4
###############################################################################


def hinge(_x):
    x, _ = unpack2D(_x)
    # x = math.pi/4.*(x)
    # y = math.pi/4.*(y)
    # return np.cos(2*x) - np.cos(y)
    return np.abs(x - 0.55)


def hill(_x):
    x, y = unpack2D(_x)
    return np.exp(-((x - .55) ** 2 + (y - .75) ** 2) / .125) + 0.01 * (x + y)


def hill_sided(_x):
    x, y = unpack2D(_x)
    center_x = 0.5
    center_y = 0.5
    sigma_1 = 1.25
    sigma_2 = 0.05
    amplitude_1 = 3.0
    amplitude_2 = 1.
    eps = 0.01
    blend_rate = 20

    delta_x = x - center_x
    delta_y = y - center_y
    alpha_x = scipy.special.expit(blend_rate * delta_x)
    # alpha_y = scipy.special.expit(blend_rate*delta_y)

    common_numerator = delta_x ** 2 + delta_y ** 2
    offset = eps * (x + y)
    h1 = amplitude_1 * np.exp(-common_numerator / sigma_1) + offset
    h2 = amplitude_2 * np.exp(-common_numerator / sigma_2) + offset
    # h3 = alpha_y*h1 + (1 - alpha_y)*h2
    # return alpha_x*h1 + (1. - alpha_x)*h3
    return alpha_x * h1 + (1. - alpha_x) * h2


def gerber(_x):
    x, y = unpack2D(_x)
    return (
        (1. / 1.) * np.exp(-((x - .25) ** 2) / 0.09)
        + (1. / 1.) * np.exp(-((y - .25) ** 2) / 0.09)
        + (1. / 3.) * np.exp(-((x - .75) ** 2) / 0.01)
        + (1. / 2.) * np.exp(-((y - .75) ** 2) / 0.01)
    )


def ranjan(_x):
    x, y = unpack2D(_x)
    return x + y + x * y


def goldsteinPrice(_x):
    x, y = unpack2D(_x)
    xa = 4 * x - 2
    ya = 4 * y - 2

    term1 = 1 + (xa + ya + 1) ** 2 * (
        19 - 14 * xa + 3 * (xa ** 2) - 14 * ya + 6 * xa * ya + 3 * (ya ** 2)
    )
    term2 = 30 + (2 * xa - 3 * ya) ** 2 * (
        18 - 32 * xa + 12 * (xa ** 2) + 48 * ya - 36 * xa * ya + 27 * (ya ** 2)
    )

    return term1 * term2


def ridge(_x):
    x, y = unpack2D(_x)
    theta = math.pi / 3.
    sigx = .05
    sigy = .04
    a = np.cos(theta) ** 2 / (2 * sigx ** 2) + np.sin(theta) ** 2 / (
        2 * sigy ** 2
    )
    b = np.sin(2 * theta) / (4 * sigx ** 2) + np.sin(2 * theta) / (
        4 * sigy ** 2
    )
    c = np.sin(theta) ** 2 / (2 * sigx ** 2) + np.cos(theta) ** 2 / (
        2 * sigy ** 2
    )

    return 0.01 * y + 0.5 * (
        np.exp(-((x - .75) ** 2) / 0.01)
        + np.exp(-((x) ** 2 + (y - 1) ** 2) / 0.1)
        + np.exp(-((x) ** 2 + (y) ** 2) / 0.005)
        - np.exp(
            -(
                a * (x - .25) ** 2
                + 2 * b * (x - .25) * (y - .25)
                + c * (y - .25) ** 2
            )
        )
    )


def ridge2(_x):
    x, y = unpack2D(_x)
    theta = math.pi / 3.
    sigx = .05
    sigy = .04
    a = np.cos(theta) ** 2 / (2 * sigx ** 2) + np.sin(theta) ** 2 / (
        2 * sigy ** 2
    )
    b = np.sin(2 * theta) / (4 * sigx ** 2) + np.sin(2 * theta) / (
        4 * sigy ** 2
    )
    c = np.sin(theta) ** 2 / (2 * sigx ** 2) + np.cos(theta) ** 2 / (
        2 * sigy ** 2
    )

    return 0.01 * y + 0.5 * (
        np.exp(-(((x - .75) ** 2) / 0.01 + ((y - .5) ** 2) / 0.4))
        + np.exp(-((x - .1) ** 2 + (y - 1) ** 2) / 0.1)
        + np.exp(-((x - .1) ** 2 + (y - .1) ** 2) / 0.005)
        - np.exp(
            -(
                a * (x - .3) ** 2
                + 2 * b * (x - .3) * (y - .25)
                + c * (y - .25) ** 2
            )
        )
    )


def test(_x):
    x, y = unpack2D(_x)
    x = 0.25 * x
    y = 0.25 * y
    return (1. / 2.) * (
        np.exp(-((x - .25) ** 2) / 0.09)
        + np.exp(-((y - .25) ** 2) / 0.09)
        + np.exp(-((x - .75) ** 2) / 0.01)
        + np.exp(-((y - .75) ** 2) / 0.01)
    )


def salomon(_x):
    x = copy.deepcopy(_x)
    if isinstance(x, np.ndarray) and x.ndim > 1:
        d = x.shape[1]
    else:
        d = len(x)

    for i in range(d):
        x[i] = 2 * x[i] - 1

    summand = 0
    for i in range(d):
        summand += x[i] ** 2
    summand = np.sqrt(summand)
    return 1 - np.cos(2 * math.pi * summand) + 0.1 * summand


def salomon2(_x):
    x = copy.deepcopy(_x)
    if isinstance(x, np.ndarray) and x.ndim > 1:
        d = x.shape[1]
    else:
        d = len(x)

    eps = 0
    for i in range(d):
        x[i] = 2 * x[i] - 1
        eps += x[i] * (i + 1)

    summand = 0
    for i in range(d):
        summand += x[i] ** 2
    summand = np.sqrt(summand)
    return 1 - np.cos(2 * math.pi * summand) + 0.1 * summand + eps


def strangulation(_x):
    x = _x[0]
    y = _x[1]
    return (
        0
        - (0.2) * np.exp(-(np.pow(x - 0.25, 2) + np.pow(y - 0.25, 2)) / 0.001)
        - (0.2) * np.exp(-(np.pow(x - 0.25, 2) + np.pow(y - 0.75, 2)) / 0.001)
        - (0.2) * np.exp(-(np.pow(x - 0.75, 2) + np.pow(y - 0.25, 2)) / 0.001)
        - (0.2) * np.exp(-(np.pow(x - 0.75, 2) + np.pow(y - 0.75, 2)) / 0.001)
        + (1.0) * np.exp(-(np.pow(x - 0.50, 2) + np.pow(y - 0.50, 2)) / 0.125)
    )
    # return (0 +
    #         (1.) * np.exp(-(np.pow(x-0.20, 2) + np.pow(y-0.20, 2))/0.005) +
    #         (1.) * np.exp(-(np.pow(x-0.80, 2) + np.pow(y-0.20, 2))/0.005) -
    #         (1.) * np.exp(-(np.pow(x-0.50, 2) + np.pow(y-0.25, 2))/0.05) -
    #         (1.) * np.exp(-(np.pow(x-0.00, 2) + np.pow(y-0.00, 2))/0.01) -
    #         (1.) * np.exp(-(np.pow(x-1.00, 2) + np.pow(y-0.00, 2))/0.01) -
    #         (1.) * np.exp(-(np.pow(x-0.00, 2) + np.pow(y-1.00, 2))/0.01) -
    #         (1.) * np.exp(-(np.pow(x-1.00, 2) + np.pow(y-1.00, 2))/0.01) +
    #         (1.) * np.exp(-(np.pow(x-0.20, 2) + np.pow(y-0.80, 2))/0.01) +
    #         (1.) * np.exp(-(np.pow(x-0.80, 2) + np.pow(y-0.80, 2))/0.01) +
    #         (1.) * np.exp(-(np.pow(x-0.50, 2) + np.pow(y-0.10, 2))/0.01) +
    #         (1.) * np.exp(-(np.pow(x-0.00, 2) + np.pow(y-0.50, 2))/0.01) +
    #         (1.) * np.exp(-(np.pow(x-1.00, 2) + np.pow(y-0.50, 2))/0.005) +
    #         (1.) * np.exp(-(np.pow(x-0.50, 2) + np.pow(y-0.90, 2))/0.01) +
    #         (1.) * np.exp(-(np.pow(x-0.45, 2) + np.pow(y-0.65, 2))/0.01) +
    #         (2.) * np.exp(-(np.pow(x-0.48, 2) + np.pow(y-0.52, 2))/0.001) +
    #         1e-2*y)


def himmelblau(_x):
    x = 12 * _x[0] - 6
    y = 12 * _x[1] - 6

    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def checkerBoard(_x):
    x = copy.deepcopy(_x)
    if isinstance(x, np.ndarray) and x.ndim > 1:
        d = x.shape[1]
    else:
        d = len(x)

    # for i in range(d):
    #     x[i] = 2*x[i] - 1

    periodicTerm = 1
    for i in range(d):
        sgn = np.cos(x[i] * math.pi) / abs(np.cos(x[i] * math.pi))
        periodicTerm *= sgn * abs(np.cos(x[i] * math.pi)) ** (1 / 7.)
    return periodicTerm


def flatTop(x):
    return checkerBoard(x) * decay(x)


def decay(_x):
    x = copy.deepcopy(_x)
    if isinstance(x, np.ndarray) and x.ndim > 1:
        d = x.shape[1]
    else:
        d = len(x)

    # for i in range(d):
    #     x[i] = 2*x[i] - 1

    eps = 0
    for i in range(d):
        eps += x[i] * 1e-6

    # decayTerm = 0
    # for i in range(d):
    #     decayTerm += x[i]**2
    # return np.exp(-decayTerm)
    dist = 0
    for i in range(d):
        dist += x[i] ** 2
    dist = np.sqrt(dist)
    if dist <= 1:
        return 0.1 + (1 - dist ** 3) ** 3 + eps
    else:
        return 0.1 + eps


def schwefel(_x):
    x = copy.deepcopy(_x)
    if isinstance(x, np.ndarray) and x.ndim > 1:
        d = x.shape[1]
    else:
        d = len(x)
    for i in range(d):
        x[i] = x[i] * 1000 - 500

    retValue = 418.9829 * d
    for i in range(d):
        retValue -= x[i] * np.sin(np.sqrt(abs(x[i])))
    return retValue


def ackley(_x):
    x = copy.deepcopy(_x)
    a = 20
    b = 0.2
    c = math.pi * 2
    if isinstance(x, np.ndarray) and x.ndim > 1:
        d = x.shape[1]
    else:
        d = len(x)
    for i in range(d):
        x[i] = x[i] * 3 - 1.5

    summand1 = 0
    summand2 = 0
    for i in range(d):
        summand1 += x[i] ** 2
        summand2 += np.cos(c * x[i])
    eps = 0
    for i in range(d):
        eps += 1e-3 * x[i]
    return (
        -a * np.exp(-b * np.sqrt(summand1 / float(d)))
        - np.exp(summand2 / float(d))
        + eps
    )


def ackley2(_x):
    x = copy.deepcopy(_x)
    a = 20
    b = 0.2
    c = math.pi * 2
    if isinstance(x, np.ndarray) and x.ndim > 1:
        d = x.shape[1]
    else:
        d = len(x)
    for i in range(d):
        x[i] = x[i] * 1.5

    summand1 = 0
    summand2 = 0
    for i in range(d):
        summand1 += x[i] ** 2
        summand2 += np.cos(c * x[i])
    eps = 0
    for i in range(d):
        eps += 1e-3 * (i + 1) * x[i]
    return (
        -a * np.exp(-b * np.sqrt(summand1 / float(d)))
        - np.exp(summand2 / float(d))
        + eps
    )


def torus(u, v):
    c = 1
    a = 0.25
    x = (c + a * np.cos(v)) * np.cos(u)
    y = (c + a * np.cos(v)) * np.sin(u)
    z = a * np.sin(v)
    return [x, y, z]


def shekel(_x):
    x = copy.deepcopy(_x)
    if isinstance(x, np.ndarray) and x.ndim > 1:
        d = x.shape[1]
    else:
        d = len(x)

    for i in range(d):
        x[i] = 10 * x[i]

    m = 4
    a = np.zeros((m, d))
    c = np.ones(m)

    # i = 0 (center of domain)
    # i = 1 (upper corner)
    # i = 2 (odd low, even high)
    # i = 3 (odd high, even low)
    c[0] = 0.25
    c[1] = 0.5
    c[2] = 0.75
    c[3] = 1
    for j in range(d):
        a[0, j] = 5
        a[1, j] = 10
        if j % 2:
            a[2, j] = 9
            a[3, j] = 1
        else:
            a[2, j] = 1
            a[3, j] = 9

    summand = 0
    for i in range(m):
        subSummand = c[i]
        for j in range(d):
            subSummand += (x[j] - a[i, j]) ** 2
        summand += (subSummand) ** -1
    return summand


def rosenbrock(_x):
    x = copy.deepcopy(_x)
    if isinstance(x, np.ndarray) and x.ndim > 1:
        d = x.shape[1]
    else:
        d = len(x)
    for i in range(d):
        x[i] = 4.8 * x[i] - 2.4

    return scipy.optimize.rosen(x)


def rosenbrock2(_x):
    x = copy.deepcopy(_x)
    if isinstance(x, np.ndarray) and x.ndim > 1:
        d = x.shape[1]
    else:
        d = len(x)
    for i in range(d):
        x[i] = 4.8 * x[i] - 2.4

    eps = 0
    return scipy.optimize.rosen(x) + eps


def gerber2(_x):
    x, y = unpack2D(_x)
    return (
        (1. / 2.) * np.exp(-((x - .25) ** 2) / 0.09)
        + (3. / 4.) * np.exp(-((x - .75) ** 2) / 0.01)
        + (1. / 1.) * np.exp(-((y - .75) ** 2) / 0.01)
    )

def gerber_different_heights(_x):
    x, y = unpack2D(_x)
    return (
        (1. / 1.) * np.exp(-((x - .25) ** 2) / 0.09)
        + (1. / 4.) * np.exp(-((y - .25) ** 2) / 0.09)
        + (1. / 3.) * np.exp(-((x - .75) ** 2) / 0.01)
        + (1. / 2.) * np.exp(-((y - .75) ** 2) / 0.01)
    )