import random 
from math import log

def log2(x):
    return log(x, 2)
    
c = [1282,1.4,2.3,5.2,3.2]
ij= [(1/3,1),(2,1),(1,1/4),(2/3,0)]

x_range = [200, 400, 600, 800, 1000]
y_range = [1,2,4,8,16]
z_range = [70,80,90]
w_range = [1,2,3]

reps=5
nparams=4
JITTER = 0.02

data_file = "input_data_" + str(nparams) + "p.txt"
outfile = open(data_file, 'w')

outfile.write("PARAMETER x\n")
if nparams > 1:
    outfile.write("PARAMETER y\n")
if nparams > 2:
    outfile.write("PARAMETER z\n")
if nparams > 3:
    outfile.write("PARAMETER w\n")
outfile.write("\n")


def eval(x):
    local_c=c[0]
    for i in range(len(c[1:])):
        local_c=local_c+c[i+1]*x[i]**ij[i][0]* log2(x[i])**ij[i][1]
    return local_c

def eval_with_noise(x):
    value = eval(x)
    rand = random.random()  # in [0.0, 1.0)
    rand = rand * 2.0 - 1.0  # scale to [-1.0, 1.0)
    return value * (1 + rand * JITTER)

for z in z_range:
    for x in x_range:
        for y in y_range:
            for w in w_range:
                outfile.write("POINTS")
                if nparams == 1:
                    outfile.write(" ( {} )".format(x))
                if nparams == 2:
                    outfile.write(" ( {} {} )".format(x, y))
                if nparams == 3:
                    outfile.write(" ( {} {} {} )".format(x, y, z))
                if nparams == 4:
                    outfile.write(" ( {} {} {} {})".format(x, y, z,w))
                outfile.write("\n")
        outfile.write("\n")
    outfile.write("\n")

outfile.write("REGION reg\n")
outfile.write("METRIC metr\n")

for z in z_range:
    for x in x_range:
        for y in y_range:
            for w in w_range:
                outfile.write("DATA")
                for i in range(1, reps + 1):
                    outfile.write(" {}".format(eval_with_noise([x,y,z,w])))
                outfile.write("\n")