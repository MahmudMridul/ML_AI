import numpy as np

# initialize an array
a = np.array([1, 2, 3, 4, 5])
a2D = np.array([[1, 2, 3], [4, 5, 6]])
a3D = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# gets dimension
dim = a2D.ndim

# gets number of rows and columns
shape = a2D.shape

# gets number of bytes one item takes in memory
item_size = a2D.itemsize

# gets total number of bytes array takes in memory
total_mem = a2D.nbytes

# access specific element
a3D[2, 0] = 100
elem = a3D[2, 0]

# gets a row
row = a3D[2, :]

# gets a column
a3D[:, 2] = np.array([10, 12, 16])
col = a3D[:, 2]

# gets elements in specified range [start_index : end_index : step]
range_elem = a3D[1:3, 1:3:2]
print(range_elem)



