import numpy as np

def distance(v1, v2, u) :
    u = np.array(u, ndmin=2)
    v = np.vstack((v1, v2))
    vv = np.dot(v, v.T) # shape (2, 2)
    uv = np.dot(u, v.T) # shape (n ,2)
    ab = np.dot(np.linalg.inv(vv), uv.T) # shape(2, n)
    w = u - np.dot(ab.T, v)
    return np.sqrt(np.sum(w**2, axis=1)) # shape (n,)

d, n = 3, 1000
v1, v2, u = np.random.rand(d), np.random.rand(d), np.random.rand(n, d)
d_ = distance(v1, v2, u)
print(d_)
# np.testing.assert_almost_equal(distance_3d(v1, v2, u), distance(v1, v2, u))
