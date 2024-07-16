import numpy as np
import matplotlib.pyplot as plt
import argparse


def himmelblau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

def load_data(filenames, xcol=0, ycol=1, **kwopts):
    xs = []
    ys = []
    for filename in filenames:
        data = np.loadtxt(filename, unpack=True, **kwopts)
        xs += list(data[xcol])
        ys += list(data[ycol])
    return np.array(xs), np.array(ys)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xcol", type=int, default=0)
    parser.add_argument("--ycol", type=int, default=1)
    parser.add_argument("--output", default="res.pdf")
    parser.add_argument("--format", default="-o")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("inputfiles", nargs="+")
    args = parser.parse_args()

    xs, ys = load_data(args.inputfiles, xcol=args.xcol, ycol=args.ycol)

    npts = 201
    c_x, c_y = np.mgrid[-6 : 6 : npts * 1j, -6 : 6 : npts * 1j]
    c_z = himmelblau(c_x, c_y)
    levels = np.logspace(0.35, 3.2, 8)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect("equal", adjustable="box")
    ax.contour(c_x, c_y, c_z, levels, colors="k")
    ax.plot(xs[args.skip : -1], ys[args.skip : -1], args.format)
    ax.set_xticks(np.linspace(-6, 6, num=5, endpoint=True))
    ax.set_yticks(np.linspace(-6, 6, num=5, endpoint=True))
    fig.savefig(args.output)
