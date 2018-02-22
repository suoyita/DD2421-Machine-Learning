import numpy as np
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# generate dataset
np.random.seed(100)
classA = np.concatenate((np.random.randn(10, 2) * 0.2 + [1.5, 0.5], np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]
inputs = np.concatenate((classA, classB))
targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))
N = inputs.shape[0]
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]


# Kernel Function
def linear_kernel(x, y):
    return np.dot(x, y)


def polynomial_kernel(x, y, p=3):
    return np.power((np.dot(x, y) + 1), p)


def radial_basis_function_kernel(x, y, sigma=1):
    diff = np.subtract(x, y)
    return math.exp((-np.dot(diff, diff)) / (2 * sigma * sigma))


P = np.zeros((N, N))
for i in range(0, N):
    for j in range(0, N):
        P[i, j] = targets[i] * targets[j] * linear_kernel([inputs[i, 0], inputs[i, 1]], [inputs[j, 0], inputs[j, 1]])


def objective(alpha):
    result_i = 0
    for i in range(0, N):
        # result_j = 0
        for j in range(0, N):
            # result_j = result_j + alpha[i] * alpha[j] * P[i][j]
            result_i = result_i + 0.5 * alpha[i] * alpha[j] * P[i][j]
    result2 = np.sum(alpha)
    return result_i - result2


def zerofun(alpha):
    return np.dot(alpha, targets)


def indicator(svs, x, y, kernel):
    ind = 0.0
    for i in range(0, len(svs)):
        ind += svs[i][0] * svs[i][3] * kernel([x, y], [svs[i][1], svs[i][2]])
    ind -= b
    return ind


def b_calculate(alpha):
    for i in range(0, N):
        if alpha[i] > 1.e-5:
            if alpha[i] < C:
                b = 0.0
                for j in range(N):
                    b += alpha[j] * targets[j] * linear_kernel(inputs[j], inputs[i])
                b -= targets[i]
                return b


start = np.zeros(N)
# alpha = np.zeros(N)
C = 10
B = [(0, C) for a in range(N)]

ret = minimize(objective, start, bounds=B, constraints={'type': 'eq', 'fun': zerofun})
alpha = ret['x']
support_vectors = []
for i in range(0, N):
    if alpha[i] > 1.e-5:
        support_vectors.append((alpha[i], inputs[i][0], inputs[i][1], targets[i]))
        plt.plot(inputs[i][0], inputs[i][1], 'g+')


b = b_calculate(alpha)

plt.plot([p[0] for p in classA],
         [p[1] for p in classA],
         'b.')
plt.plot([p[0] for p in classB],
         [p[1] for p in classB],
         'r.')
plt.axis('equal')
plt.savefig('svmplot.pdf')
# plt.show()

xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)

grid = np.array([[indicator(support_vectors, x, y, linear_kernel)
                  for x in xgrid]
                 for y in ygrid])

plt.contour(xgrid, ygrid, grid,
            (-1.0, 0.0, 1.0),
            colors=('red', 'black', 'blue'),
            linewidths=(1, 3, 1))
plt.show()
