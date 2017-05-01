import csv

import numpy as np
import scipy
import sympy
from scipy.optimize import lsq_linear
from sklearn.linear_model import LogisticRegression
#P contains the input in the columns as array
# x is a row array
def one_cara(x, P):
    A = np.vstack((P, np.ones((P.shape[1]))))
    b = np.hstack((x, [1]))
    lamb = lsq_linear(A, b, bounds=(0, np.inf)).x

    B = (P.transpose() - P.transpose()[0])[1:, :]
    B = B.transpose()
    mu = np.array(sympy.Matrix(B).nullspace())[0, :]# lsq_linear(B, c).x
    mu1 = np.array([-1 * np.sum(mu)])
    mu = np.hstack((mu1, mu))
    gz = np.where(mu > 0)
    alpha = min(lamb[gz] / mu[gz])
    new_coeff = lamb - alpha * mu
    return new_coeff

#P contains the input in the columns
def cara(x, P):
    for i in range(P.shape[1] - P.shape[0] - 1):
        c = one_cara(x, np.array(P))
        j = np.argmin(np.abs(c))
        P = np.delete(P, j, 1)
    return P

def mvee(points, tol = 0.001):
    """
    Find the minimum volume ellipse.
    Return A, c where the equation for the ellipse given in "center form" is
    (x-c).T * A * (x-c) = 1  ->c should be zero
    """
    points = np.asmatrix(points)
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = Q * np.diag(u) * Q.T
        M = np.diag(Q.T * np.linalg.inv(X) * Q)
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = np.linalg.norm(new_u-u)
        u = new_u
    c = u*points
    A = np.linalg.inv(points.T*np.diag(u)*points - c.T*c)/d
    return np.asarray(A), np.squeeze(np.asarray(c))

#P contains the input in the columns
def infty_coreset(P, k):
    D = np.matrix(np.column_stack((P, -P)))
    A, c = mvee(D.transpose())
    tau = np.linalg.cholesky(np.linalg.inv(A)) #tau
    t = np.linalg.inv(tau)
    N = (t * D )/ k
    d = P.shape[0]
    B = np.identity(d)
    Q = np.empty((d, 2 * (d + 1)*d))
    for i in range(d):
        e = B[i, :]
        j = i * 6
        Q[:, j: j + d + 1] = cara(e, N)
        Q[:, j + d + 1 : j + 2 * (d + 1)] = cara(-1 * e, N)
    Q = np.array(tau * np.matrix(Q))
    aset = set([tuple(x) for x in Q.transpose()])
    bset = set([tuple(x) for x in P.transpose()])
    return np.array([x for x in aset & bset])

    #need that the min-norm is bigger then


P = np.array([
    [2, 0],
    [-2, 0],
    [0, 2],
    [0, -2],
    [1, 1],
    [-0.5, 0.5]
])
# print(P)
# infty_coreset(P.transpose(), 1)
# print(cara(np.array([1, 1]), P.transpose()))
reactions = np.genfromtxt('ds1.100.csv', delimiter=',')
y = reactions[:, 100]
x = reactions[:, :100]
Q = infty_coreset(x, 1)
print(Q.shape)
# clf = LogisticRegression(penalty='l1')
# clf.fit(x, y)
# pred = clf.predict(x)
# print(pred)
# diff = 0
# for p, l in zip(pred, y):
#     if p != l:
#         diff += 1
# print(len(y))
# print(str(diff/len(y)))
# print(clf.score(x, y))


