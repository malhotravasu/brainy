import numpy as np
# import math

# def signum(data):
#     return math.copysign(1, data)

def sigmoid(data):
    return 1/(1+np.exp(-data))


