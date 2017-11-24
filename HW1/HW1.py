import numpy as np
import matplotlib.pyplot as plt
import time


# James Heinlein


def center(D):
    mean = D.mean(axis=0)
    Z = D - mean
    return Z


def cov1(Z):
    # element-wise definition of sample covariance
    S = np.zeros((Z.shape[0], Z.shape[0]))
    for i in range(0, Z.shape[0]):
        column_i = Z[i, :]

        for j in range(0, Z.shape[0]):
            column_j = Z[j, :]
            s = np.dot(column_i, column_j)
            s /= (Z.shape[1] - 1)
            S[i, j] = s
    return S

def cov2(Z):
    # Matrix product
    n = Z.shape[1]
    result = Z @ Z.T

    return result / n


def cov3(Z):
    # Sum of outer products

    n = Z.shape[1]
    S = np.zeros((Z.shape[0], Z.shape[0]))
    for i in range(0, Z.shape[1]):
        S += np.outer(Z[:, i], Z[:,i])
    S /= Z.shape[1]

    return S


def calculate_r(D, a):

    #Calculate variance explained by each PC
    a *= 100
    eigvals = np.linalg.eigvals(D)
    total = sum(eigvals)
    var_exp = [(i/total) * 100 for i in sorted(eigvals, reverse=True)]

    #iterate over explained variance of each eigenvalue
    itr = iter(var_exp)
    var_sum = 0
    r = 0
    while var_sum < a:
        var_sum += next(itr)
        r += 1

    return r



# Center data and compute covariance matrix
data = np.genfromtxt("cloud data DB1.tsv")
centered = center(data)
covariance = cov2(centered.T)


#Calculate r components needed for 90% retained variance
print(calculate_r(covariance, .90), " components needed for 90% retained variance")

#test cov1()
t0 = time.time()
Y1 = cov1(centered)
t1 = time.time()
print("cov1() took %.6f seconds" % (t1-t0))

#test cov2()
t0 = time.time()
Y2 = cov2(centered)
t1 = time.time()
print("cov2() took %.6f seconds" % (t1-t0))

# test cov3()
t0 = time.time()
Y3 = cov3(centered)
t1 = time.time()
print("cov3() took %.6f seconds" % (t1-t0))

#test cov1()
t0 = time.time()
Y1 = np.cov(centered)
t1 = time.time()
print("numpy.cov() took %.6f seconds" % (t1-t0))

# compute eigenpairs
evals, evecs = np.linalg.eig(covariance)

# Sort the pairs by eigenvalue
index = evals.argsort()[::-1]
evals = evals[index]
evecs = evecs[:, index]

# zip the pairs together
epairs = [(evals[i], evecs[:, i]) for i in range(len(evals))]

# compute transformation matrix
W = np.hstack((epairs[0][1].reshape(10,1),
               epairs[1][1].reshape(10,1)))


# project points into new basis
Y = centered.dot(W).T
plt.title('Reduced Cloud data')
plt.scatter(Y[0], Y[1])
plt.show()


# Select PC1 and PC2 to plot
pc1 = epairs[0]
pc2 = epairs[1]


# make matrix of unit vectors
U = np.eye(10)

X = [1,2,3,4,5,6,7,8,9,10]
Y1 = []
Y2 = []


# create PC1 points
for i in range(0,10):
    Y1.append(np.dot(pc1[1], U[i]))

# create PC2 points
for i in range(0,10):
    Y2.append(np.dot(pc2[1], U[i]))

plt.title('PC1 and PC2: magnitude in each original dimension')
plt.plot(X, Y1, color='red', label='PC1')
plt.plot(X, Y2, color='blue', label='PC2')
plt.legend()
plt.show()




