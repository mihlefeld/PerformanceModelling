import random
from itertools import product
from math import log

c  = [1282, 1.4, 2.3, 5.2, 3.2, 4.2]
ij = [(1/3, 1), (2, 1), (1, 1/4), (2/3, 0), (1/3, 1)]
JITTER = 0.02

def log2(x):
    return log(x, 2)


def eval(x):
    local_c = c[0]
    for i in range(len(c[1:])):
        local_c += c[i+1] * x[i]**ij[i][0] * log2(x[i])**ij[i][1]
    return local_c


def eval_with_noise(x):
    value = eval(x)
    rand = random.random()  # in [0.0, 1.0)
    rand = rand * 2.0 - 1.0  # scale to [-1.0, 1.0)
    return value * (1 + rand * JITTER)


def main():
    x_range = [200, 400, 600, 800, 1000]
    y_range = [1, 2, 4, 8, 16]
    z_range = [70, 80, 90]
    w_range = [1, 2, 3]
    v_range = [4, 5]

    reps = 5
    nparams = 5

    name = ['one', 'two', 'three', 'four', 'five']

    data_file = f"../tests/{name[nparams-1]}_parameter_1_converted.txt"
    outfile = open(data_file, 'w')
    all_params = list(product(*[x_range, y_range, z_range, w_range, v_range][:nparams]))
    outfile.write(f"extrap measurements {nparams} {len(all_params)}\n")

    for params in all_params:
        outfile.write(" ".join([str(p) for p in params] + [str(eval_with_noise(params))]))
        outfile.write("\n")


if __name__ == '__main__':
    main()