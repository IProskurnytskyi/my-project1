import sys
import numpy as np
from numpy import pi, newaxis

a = np.array([1, 2, 3, 4, 5])
b = np.array([0, .5, 1, 1.5, 2])

print(a[::2])
print(a[::-2])
print(b[[0, 2, 1]])

print(a.dtype)
print(b.dtype)

c = np.array([1, 2, 3, 4], dtype=float)
print(c)

c = np.array([1, 2, 3, 4], dtype=str)
print(c)

c = np.array([[1, 2], [3, 4]], dtype=complex)
print(c)

print(np.zeros((3, 4)))

print(np.ones((2, 3, 4), dtype=np.int16)) # 2 means two arrays of size 3x4

print(np.empty((2, 3))) # empty() creates an array whose initial content is random and depends on the state of the memory.

print(np.arange(10, 30, 5)) # arange() the same like range() but returns an array.

print(np.arange(0, 2, 0.3))  # It accepts float arguments.

print(np.linspace(0, 2, 9)) # in range 9 numbers from 0 to 2 instead of steps.

x = np.linspace(0, 2 * pi, 20) # Useful to evaluate function at lots of points.
print(x)

f = np.sin(x)
print(f)

print("-"*100)

a = np.array([20, 30, 40, 50])
print(10 * np.sin(a))
print(a < 35)

g = np.arange(24).reshape(2, 3, 4)  # 3d array. 2 means two matrices.
print(g)

print(np.arange(10000)) # If an array is too large to be printed, NumPy automatically skips the central part of the array and only prints the corners.

print(np.arange(10000).reshape(100, 100))

# np.set_printoptions(threshold=sys.maxsize) # To disable this behaviour and force NumPy to print the entire array,
# print(np.arange(10000).reshape(100, 100)) # you can change the printing options using set_printoptions.

A = np.array([[1, 1],
              [0, 1]])
B = np.array([[2, 0],
              [3, 4]])

print(A * B) # elementwise product, so we just * them.
print(A @ B) # matrix product, so we * and + them.
print(A.dot(B)) # another matrix product.

print("-"*100)

rg = np.random.default_rng(1)  # create instance of default random number generator
a = np.ones((2, 3), dtype=int)
b = rg.random((2, 3))
a *= 3 # Some operations, such as += and *=, act in place to modify an existing array rather than create a new one.
b += a

print(a)
print(b)

a = np.ones(3, dtype=np.int32)
b = np.linspace(0, pi, 3)
print(a)
print(b)
print(b.dtype.name)

c = a + b
print(c)

d = np.exp(c * 1j)
print(d)
print(d.dtype.name)

a = rg.random((2, 3))
print(a)
print(a.sum())
print(a.min())
print(a.max())

b = np.arange(12).reshape(3, 4)
print(b)
print(b.sum(axis=0)) # sum of each column
print(b.min(axis=1)) # min of each row
print(b.cumsum(axis=1)) # cumulative sum along each row, so next value plus previous in the row.

print("-"*100)

B = np.arange(3)
print(B)
print(np.exp(B))
print(np.sqrt(B))

C = np.array([2., -1., 4.])
print(np.add(B, C))

a = np.arange(10)**3
print(a)
a[:6:2] = 1000 # from start to position 6, exclusive, set every 2nd element to 1000.
print(a)
print(a[::-1]) # reversed a

for i in a:
    print(i**(1 / 3))

print("-"*100)

def f(x, y):
    return 10 * x + y
# print(f(2, 4))

b = np.fromfunction(f, (5, 4), dtype=int)
print(b)
print(b[0:5, 1]) # each row in the second column of b
print(b[:, 1]) # the same
print(b[1:3, :])
print(b[-1]) # When fewer indices are provided than the number of axes, the missing indices are considered complete slices.

c = np.array([[[  0,  1,  2], # a 3D array (two stacked 2D arrays)
               [ 10, 12, 13]],
              [[100, 101, 102],
               [110, 112, 113]]])

print(c.shape)
print(c[1, ...]) # same as c[1, :, :] or c[1]
print(c[..., 2]) # same as c[:, :, 2]

for row in b:
    print(row)

for element in b.flat:
    print(element) 

a = np.floor(10 * rg.random((3, 4)))
print(a)
print(a.ravel())
print(a.reshape(6, 2))
print(a.T) # returns the array, transposed.
print(a.T.shape)
print(a.shape)
print(a.reshape(3, -1)) # If a dimension is given as -1 in a reshaping operation, the other dimensions are automatically calculated.

print("-"*100)

a = np.floor(10 * rg.random((2, 2)))
b = np.floor(10 * rg.random((2, 2)))
print(np.vstack((a, b)))
print(np.hstack((a, b)))
print(np.column_stack((a, b))) # The function column_stack stacks 1D arrays as columns into a 2D array. It is equivalent to hstack only for 2D arrays.

a = np.array([4., 2.])
b = np.array([3., 8.])
print(np.column_stack((a, b))) # returns a 2D array.
print(np.hstack((a, b))) # the result is different.
print(a[:, newaxis]) # View 'a' as a 2D column vector.
print(np.column_stack((a[:, newaxis], b[:, newaxis])))
print(np.hstack((a[:, newaxis], b[:, newaxis])))  # the result is the same. 

print(np.column_stack is np.hstack)
print(np.row_stack is np.vstack)

print(np.r_[1:4, 0, 4]) # In complex cases, r_ and c_ are useful for creating arrays by stacking numbers along one axis. They allow the use of range literals

a = np.floor(10 * rg.random((2, 12)))
print(a)
print(np.hsplit(a, 3)) # Split 'a' into 3.
print(np.hsplit(a, (3, 4))) # Split 'a' after the third and the fourth column.

x = np.arange(8.0)
print(np.array_split(x, 3))
x = np.arange(9)
print(np.array_split(x, 4))

print("-"*100)

a = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])

def f(x):
    return id(x)
print(id(a))
print(f(a))

c = a.view()
print(c is a)
print(c.base is a)
print(c.flags.owndata)

s = a[:, 1:3]
s[:] = 10 # s[:] is a view of s. Note the difference between s = 10 and s[:] = 10
print(a)

d = a.copy
print(d is a)



















































































