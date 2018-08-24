import math
import time
import scipy
import numpy as np
import pandas as pd

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import LightSource
from matplotlib.markers import MarkerStyle

import seaborn as sns
import topopy

from matplotlib.colors import LinearSegmentedColormap

from test_functions import *

magma_cmap = matplotlib.cm.get_cmap('magma')
viridis_cmap = matplotlib.cm.get_cmap('viridis')
cividis_cmap = matplotlib.cm.get_cmap('cividis')
earth_cmap = matplotlib.cm.get_cmap('gist_earth')
terrain_cmap = matplotlib.cm.get_cmap('terrain')
set3_cmap = matplotlib.cm.get_cmap('Set3')

unique_colors = 12
set3 = []
for i in range(unique_colors):
    set3.append([float(i)/unique_colors,
                 'rgb({}, {}, {})'.format(*set3_cmap(i))])
    set3.append([float(i+1)/unique_colors,
                 'rgb({}, {}, {})'.format(*set3_cmap(i))])


def color_msc(X, Y, p):
    msc = topopy.MorseSmaleComplex(max_neighbors=8, graph='beta skeleton')
    msc.build(X, Y)

    partitions = msc.get_partitions(p)
    unique_keys = partitions.keys()
    encoded_keys = {}
    next_value = 0

    for key in unique_keys:
        encoded_keys[key] = next_value
        next_value = (next_value+1) % unique_colors

    C = np.zeros(len(X))
    for key, items in partitions.items():
        C[list(items)] = encoded_keys[key]

    return C


min_x = 0
max_x = 1
resolution = 500
p = 5e-2
max_steps = 10000
stopping_crieterion = 1e-6
push_size = 1
step_size = 0.005
test_function = df
# test_function = ackley


def dxdy(x, foo=test_function):
    h = 1e-3
    dx0 = (foo(x+np.array([h, 0])) - foo(x)) / h
    dx1 = (foo(x+np.array([0, h])) - foo(x)) / h
    grad_x = np.array([dx0, dx1]).flatten()
    mag = np.linalg.norm(grad_x)
    if mag < stopping_crieterion:
        return np.array([0, 0]).flatten()
    return grad_x / mag * step_size


x, y = np.mgrid[min_x:max_x:(resolution * 1j), min_x:max_x:(resolution * 1j)]
X = np.vstack([x.ravel(), y.ravel()]).T

Z = np.empty(X.shape[0])
for i, xi in enumerate(X):
    Z[i] = test_function(xi)
z = Z.reshape(x.shape)

start = time.time()
msc = topopy.MorseSmaleComplex(max_neighbors=8, graph='beta skeleton')
msc.build(X, Z)
msc.set_persistence(p)
end = time.time()
print('Build MSC: {} s'.format(end-start))
start = time.time()

mins = []
maxs = []
saddles = []
saddle_ptrs = {}
for key in msc.get_partitions().keys():
    mins.append(key[0])
    maxs.append(key[1])
for line in msc.print_hierarchy().strip().split(' '):
    tokens = line.split(',')
    if p <= float(tokens[1]):
        saddles.append(int(tokens[-1]))
        if saddles[-1] not in saddle_ptrs:
            saddle_ptrs[saddles[-1]] = []
        saddle_ptrs[saddles[-1]].append(int(tokens[2]))
        saddle_ptrs[saddles[-1]].append(int(tokens[3]))

# [116,
#  104293,
#  128500,
#  ,
#  130423,
#  163592,
#  ,
#  189235,
#  207588,
#  229500,
#  249748]
saddles.remove(129428)
saddles.remove(188237)

for m in mins+maxs:
    if m in saddles:
        saddles.remove(m)

C = color_msc(X, Z, p)
c = C.reshape(z.shape)

end = time.time()
print('Extract Extrema and Color: {} s'.format(end-start))

start = time.time()
plt.figure(num=None, figsize=(8, 8), dpi=100, facecolor='w')

plt.scatter(X[:, 0], X[:, 1], c=C / np.max(C),
            cmap=set3_cmap, zorder=1, s=1, marker=',')

def trace_path(current_x, sgn=1):
    grad_x = dxdy(current_x)
    steps = 0
    trace = [np.array(current_x)]
    while np.linalg.norm(grad_x) > stopping_crieterion and steps < max_steps and current_x[0] > -1 and current_x[0] < 2 and current_x[1] > -1 and current_x[1] < 2:
        grad_x = sgn * dxdy(current_x)
        current_x += grad_x
        trace.append(np.array(current_x))
        steps += 1
    return np.array(trace)

traces = []
traces.append(trace_path(np.array([0.81, 0.18])))

for saddle in sorted(saddles):
    for i in saddle_ptrs[saddle]:
        current_x = np.array(X[saddle])
        d1 = X[i] - current_x
        d1 *= step_size / np.linalg.norm(d1)
        current_x += d1
        traces.append(trace_path(current_x))

        current_x = np.array(X[saddle])
        d1 = X[i] - current_x
        d1 *= step_size / np.linalg.norm(d1)
        current_x += d1
        traces.append(trace_path(current_x, -1))

    # Positive gradient
    current_x = np.array(X[saddle])
    traces.append(trace_path(current_x))

    # Positive gradient, but take a step in the opposite
    # direction to head toward the other maximum
    current_x = np.array(X[saddle])
    grad_x = -1 * dxdy(current_x)
    current_x += push_size*grad_x
    traces.append(trace_path(current_x))

    # Negative gradient
    current_x = np.array(X[saddle])
    traces.append(trace_path(current_x, -1))

    # Negative gradient, but take a step in the opposite
    # direction to head toward the other minimum
    current_x = np.array(X[saddle])
    grad_x = dxdy(current_x)
    current_x += push_size*grad_x
    traces.append(trace_path(current_x, -1))

end = time.time()
print('Tracing from saddles: {} s'.format(end-start))

start = time.time()

lw = 1
cps = {
    'min': mins,
    'max': maxs,
    'saddle': saddles}
colors = {
    'min': "#377eb8",
    'max': "#e41a1c",
    'saddle': "#4daf4a"}

for name in ['min', 'max', 'saddle']:
    idxs = cps[name]
    color = colors[name]
    plt.scatter(X[idxs, 0], X[idxs, 1], s=50, c=color,
                edgecolor='k', linewidth=lw, zorder=8)

for trace in traces:
    plt.plot(trace[:, 0], trace[:, 1], c="#000000", linewidth=5, zorder=6)
    plt.plot(trace[:, 0], trace[:, 1], c="#984ea3", linewidth=3, zorder=7)

# plt.contourf(x,y,z, cmap=plt.cm.gist_earth)
lws = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2]
# levels = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1]
# print(min(Z), max(Z))
plt.contourf(x, y, z, cmap=plt.cm.gist_gray, alpha=0.2, linewidths=lws,
             vmin=np.min(Z), vmax=np.max(Z), zorder=2)
plt.contour(x, y, z, colors='k', alpha=0.5, linewidths=lws,
            linestyles='solid', zorder=3)

# valid_saddles = []
# for saddle in saddles:
#     if X[saddle, 0] > 0 and X[saddle, 0] < 1 and X[saddle, 1] > 0 and X[saddle, 1] < 1:
#         valid_saddles.append(saddle)
# valid_saddles = [7348, 5183, 8119, 6617, 7647, 5283, 4158, 5100]
# print(Z[4158], (Z[7647] + Z[5183]) / 2, Z[6617], Z[8119])
# level_sets = [Z[6617]]
# level_sets = [0.52]


# plt.contour(x, y, z, colors='#000000', linewidths=4, linestyles='solid', levels=sorted(level_sets), zorder=2)
# plt.contour(x, y, z, colors='#ff7f00', linewidths=2, linestyles='solid', levels=sorted(level_sets), zorder=3)

print(saddles)

plt.contour(x, y, z, colors='#000000', linewidths=4,
            linestyles='solid', levels=[0.52], zorder=4)
plt.contour(x, y, z, colors='#ff7f00', linewidths=2,
            linestyles='solid', levels=[0.52], zorder=5)

start_x = 0
end_x = int(1*resolution)
start_y = int(0*resolution)
end_y = int(0.4*resolution)
plt.contour(x[start_x:end_x, start_y:end_y], y[start_x:end_x, start_y:end_y], z[start_x:end_x, start_y:end_y],
            colors='#000000', linewidths=4, linestyles='solid', levels=[0.27534601910957013], zorder=4)
plt.contour(x[start_x:end_x, start_y:end_y], y[start_x:end_x, start_y:end_y], z[start_x:end_x, start_y:end_y],
            colors='#ff7f00', linewidths=2, linestyles='solid', levels=[0.27534601910957013], zorder=5)

start_x = 0
end_x = int(0.6*resolution)
start_y = int(0.4*resolution)
end_y = int(0.8*resolution)
plt.contour(x[start_x:end_x, start_y:end_y], y[start_x:end_x, start_y:end_y], z[start_x:end_x, start_y:end_y],
            colors='#000000', linewidths=4, linestyles='solid', levels=[1.1566348467148404], zorder=4)
plt.contour(x[start_x:end_x, start_y:end_y], y[start_x:end_x, start_y:end_y], z[start_x:end_x, start_y:end_y],
            colors='#ff7f00', linewidths=2, linestyles='solid', levels=[1.1566348467148404], zorder=5)

start_x = int(0.6*resolution)
end_x = int(0.9*resolution)
start_y = int(0.1*resolution)
end_y = int(0.4*resolution)
plt.contour(x[start_x:end_x, start_y:end_y], y[start_x:end_x, start_y:end_y], z[start_x:end_x, start_y:end_y],
            colors='#000000', linewidths=4, linestyles='solid', levels=[0.07005384273167777], zorder=4)
plt.contour(x[start_x:end_x, start_y:end_y], y[start_x:end_x, start_y:end_y], z[start_x:end_x, start_y:end_y],
            colors='#ff7f00', linewidths=2, linestyles='solid', levels=[0.07005384273167777], zorder=5)
plt.axes().set_aspect('equal')
plt.axis('off')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)

end = time.time()
print('Plot: {} s'.format(end-start))

plt.show()
