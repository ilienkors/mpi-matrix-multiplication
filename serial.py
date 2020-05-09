import time
from functions import get_matrix

def matrixMult(a, b, res):
    time_start = time.time()
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                res[i][j] += a[i][k] * b[k][j]
    print("Finished in %s seconds" % (time.time() - time_start))



sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384]

print("SERIAL")

for size in sizes:
    first_matrix = get_matrix(size, 36.6)
    second_matrix = get_matrix(size, 1.23)
    res = get_matrix(size, 0)
    matrixMult(first_matrix, second_matrix, res)