import math
import time
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import json

import topopy

set3_cmap = matplotlib.cm.get_cmap("Set3")

unique_colors = 12
set3 = []
for i in range(unique_colors):
    set3.append([float(i) / unique_colors, "rgb({}, {}, {})".format(*set3_cmap(i))])
    set3.append([float(i + 1) / unique_colors, "rgb({}, {}, {})".format(*set3_cmap(i))])


def color_msc(X, Y, p):
    msc = topopy.MorseSmaleComplex(max_neighbors=8, graph="beta skeleton")
    msc.build(X, Y)

    partitions = msc.get_partitions(p)
    unique_keys = partitions.keys()
    encoded_keys = {}
    next_value = 0

    for key in unique_keys:
        encoded_keys[key] = next_value
        next_value = (next_value + 1) % unique_colors

    C = np.zeros(len(X))
    for key, items in partitions.items():
        C[list(items)] = encoded_keys[key]

    return C


def bump(x, y, amplitude=1.0 / 4.0, cx=0.5, cy=0.5):
    return amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 0.001)


def unpack2D(_x):
    """
    Helper function for splitting 2D data into x and y component to make
    equations simpler
    """
    _x = np.atleast_2d(_x)
    x = _x[:, 0]
    y = _x[:, 1]
    return x, y


def df(_x):
    p = _x
    centers = [
        np.array([0.801570375639, 0.161880925191]),
        np.array([0.829773664309, 0.225535923249]),
        np.array([0.126453126536, 0.982384428954]),
        np.array([0.693347266615, 0.764874190406]),
        np.array([0.415066271455, 0.181807048897]),
    ]
    powers = [
        1.17269364563,
        1.38511464016,
        1.93062086148,
        1.71429326706,
        1.7429854611,
    ]

    covars = []
    covars.append(
        np.array(
            [
                [6.861127634772008044e00, 5.713205235351410671e00],
                [5.713205235351410671e00, 1.182733672915492917e01],
            ]
        )
    )
    covars.append(
        np.array(
            [
                [1.832786696485626265e01, 8.906194773508316231e00],
                [8.906194773508316231e00, 1.072361068483539803e01],
            ]
        )
    )
    covars.append(
        np.array(
            [
                [5.039436493504411807e00, 4.461204945275424549e00],
                [4.461204945275424549e00, 1.137522966049168183e01],
            ]
        )
    )
    covars.append(
        np.array(
            [
                [2.087529540101837000e01, 7.181066748222261431e00],
                [7.181066748222261431e00, 8.028676705850410045e00],
            ]
        )
    )
    covars.append(
        np.array(
            [
                [4.053108367744338913e00, 2.245412739483759967e00],
                [2.245412739483759967e00, 1.817496301681578785e01],
            ]
        )
    )

    dist = [
        math.pow(math.sqrt(np.dot(p - c, np.dot(covar, (p - c)))), power)
        for c, covar, power in zip(centers, covars, powers)
    ]

    min_dist = min(dist)
    x, y = unpack2D(_x)
    return (
        min_dist
        + bump(x, y, amplitude=1.0, cx=1, cy=1)
        + bump(x, y, amplitude=1.0, cx=0, cy=0.6)
        + bump(x, y, amplitude=1.0, cx=0.7, cy=0.0)
        + bump(x, y, amplitude=1.0, cx=0.494987, cy=0.581399)
        + bump(x, y, amplitude=1.0, cx=1, cy=0.41)
        + bump(x, y, amplitude=1.0, cx=1, cy=0.0)
    )


min_x = 0
max_x = 1
resolution = 500
p = 5e-2
max_steps = 10000
stopping_crieterion = 1e-6
push_size = 1
step_size = 0.005
test_function = df


def dxdy(x, foo=test_function):
    h = 1e-3
    dx0 = (foo(x + np.array([h, 0])) - foo(x)) / h
    dx1 = (foo(x + np.array([0, h])) - foo(x)) / h
    grad_x = np.array([dx0, dx1]).flatten()
    mag = np.linalg.norm(grad_x)
    if mag < stopping_crieterion:
        return np.array([0, 0]).flatten()
    return grad_x / mag * step_size


x, y = np.mgrid[min_x : max_x : (resolution * 1j), min_x : max_x : (resolution * 1j)]
X = np.vstack([x.ravel(), y.ravel()]).T

Z = np.empty(X.shape[0])
for i, xi in enumerate(X):
    Z[i] = test_function(xi)
z = Z.reshape(x.shape)

start = time.time()
msc = topopy.MorseSmaleComplex(max_neighbors=8, graph="beta skeleton")
msc.build(X, Z)
msc.set_persistence(p)
end = time.time()
print("Build MSC: {} s".format(end - start))
start = time.time()

mins = []
maxs = []
saddles = []
saddle_ptrs = {}
for key in msc.get_partitions().keys():
    mins.append(key[0])
    maxs.append(key[1])

json_object = json.loads(msc.to_json())
for merge in json_object["Hierarchy"]:
    if p <= merge["Persistence"]:
        saddles.append(merge["Saddle"])
        if saddles[-1] not in saddle_ptrs:
            saddle_ptrs[saddles[-1]] = []
        saddle_ptrs[saddles[-1]].append(merge["Dying"])
        saddle_ptrs[saddles[-1]].append(merge["Surviving"])

saddles.remove(129428)
saddles.remove(188237)

for m in mins + maxs:
    if m in saddles:
        saddles.remove(m)

C = color_msc(X, Z, p)
c = C.reshape(z.shape)
C /= np.max(C)

end = time.time()
print("Extract Extrema and Color: {} s".format(end - start))

start = time.time()
plt.figure(num=None, figsize=(8, 8), dpi=100, facecolor="w")

plt.scatter(X[:, 0], X[:, 1], c=C, cmap=set3_cmap, zorder=1, s=1, marker=",")


def trace_path(current_x, sgn=1):
    grad_x = dxdy(current_x)
    steps = 0
    trace = [np.array(current_x)]
    while (
        np.linalg.norm(grad_x) > stopping_crieterion
        and steps < max_steps
        and current_x[0] > -1
        and current_x[0] < 2
        and current_x[1] > -1
        and current_x[1] < 2
    ):
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
    current_x += push_size * grad_x
    traces.append(trace_path(current_x))

    # Negative gradient
    current_x = np.array(X[saddle])
    traces.append(trace_path(current_x, -1))

    # Negative gradient, but take a step in the opposite
    # direction to head toward the other minimum
    current_x = np.array(X[saddle])
    grad_x = dxdy(current_x)
    current_x += push_size * grad_x
    traces.append(trace_path(current_x, -1))

end = time.time()
print("Tracing from saddles: {} s".format(end - start))

start = time.time()

lw = 1
cps = {"min": mins, "max": maxs, "saddle": saddles}
colors = {"min": "#377eb8", "max": "#e41a1c", "saddle": "#4daf4a"}

for name in ["min", "max", "saddle"]:
    idxs = cps[name]
    color = colors[name]
    plt.scatter(
        X[idxs, 0],
        X[idxs, 1],
        s=50,
        c=color,
        edgecolor="k",
        linewidth=lw,
        zorder=8,
    )

for trace in traces:
    plt.plot(trace[:, 0], trace[:, 1], c="#000000", linewidth=5, zorder=6)
    plt.plot(trace[:, 0], trace[:, 1], c="#984ea3", linewidth=3, zorder=7)

lws = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2]
plt.contourf(
    x,
    y,
    z,
    cmap=plt.cm.gist_gray,
    alpha=0.2,
    linewidths=lws,
    vmin=np.min(Z),
    vmax=np.max(Z),
    zorder=2,
)
plt.contour(x, y, z, colors="k", alpha=0.5, linewidths=lws, zorder=3)

print(saddles)

plt.contour(
    x,
    y,
    z,
    colors="#000000",
    linewidths=4,
    linestyles="solid",
    levels=[0.52],
    zorder=4,
)
plt.contour(
    x,
    y,
    z,
    colors="#ff7f00",
    linewidths=2,
    linestyles="solid",
    levels=[0.52],
    zorder=5,
)

start_x = 0
end_x = int(1 * resolution)
start_y = int(0 * resolution)
end_y = int(0.4 * resolution)
plt.contour(
    x[start_x:end_x, start_y:end_y],
    y[start_x:end_x, start_y:end_y],
    z[start_x:end_x, start_y:end_y],
    colors="#000000",
    linewidths=4,
    linestyles="solid",
    levels=[0.27534601910957013],
    zorder=4,
)
plt.contour(
    x[start_x:end_x, start_y:end_y],
    y[start_x:end_x, start_y:end_y],
    z[start_x:end_x, start_y:end_y],
    colors="#ff7f00",
    linewidths=2,
    linestyles="solid",
    levels=[0.27534601910957013],
    zorder=5,
)

start_x = 0
end_x = int(0.6 * resolution)
start_y = int(0.4 * resolution)
end_y = int(0.8 * resolution)
plt.contour(
    x[start_x:end_x, start_y:end_y],
    y[start_x:end_x, start_y:end_y],
    z[start_x:end_x, start_y:end_y],
    colors="#000000",
    linewidths=4,
    linestyles="solid",
    levels=[1.1566348467148404],
    zorder=4,
)
plt.contour(
    x[start_x:end_x, start_y:end_y],
    y[start_x:end_x, start_y:end_y],
    z[start_x:end_x, start_y:end_y],
    colors="#ff7f00",
    linewidths=2,
    linestyles="solid",
    levels=[1.1566348467148404],
    zorder=5,
)

start_x = int(0.6 * resolution)
end_x = int(0.9 * resolution)
start_y = int(0.1 * resolution)
end_y = int(0.4 * resolution)
plt.contour(
    x[start_x:end_x, start_y:end_y],
    y[start_x:end_x, start_y:end_y],
    z[start_x:end_x, start_y:end_y],
    colors="#000000",
    linewidths=4,
    linestyles="solid",
    levels=[0.07005384273167777],
    zorder=4,
)
plt.contour(
    x[start_x:end_x, start_y:end_y],
    y[start_x:end_x, start_y:end_y],
    z[start_x:end_x, start_y:end_y],
    colors="#ff7f00",
    linewidths=2,
    linestyles="solid",
    levels=[0.07005384273167777],
    zorder=5,
)
plt.axes().set_aspect("equal")
plt.axis("off")
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)

end = time.time()
print("Plot: {} s".format(end - start))

plt.show()
