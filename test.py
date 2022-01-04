import numpy as np

def shuffle_rowwise(arr):
    for i in range(len(arr)):
        np.random.shuffle(arr[i])
    return arr

arr = np.arange(10)
arr = arr.reshape((5,2))

arr = shuffle_rowwise(arr).T
print(arr)
