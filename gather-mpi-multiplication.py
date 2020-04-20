import time
from mpi4py import MPI
from functions import line_and_second_matrix_multiplication
from functions import get_matrix

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
num_of_processes = comm.Get_size()

second_matrix = get_matrix(num_of_processes, 36.6)

if my_rank == 0:
    first_matrix = get_matrix(num_of_processes, 1.23)
    first_matrix_row = [x for x in first_matrix]
else:
    first_matrix_row = None

if my_rank == 0:
    time_start = time.time()

first_matrix_row = comm.scatter(first_matrix_row, root=0)

line = [my_rank, line_and_second_matrix_multiplication(
    first_matrix_row, second_matrix)]

data = comm.gather(line, root=0)

if my_rank == 0:
    result_matrix = [[] for i in range(0, num_of_processes)]
    for row in data:
        result_matrix[row[0]] = row[1]
    # print(result_matrix)

if my_rank == 0:
    print("Finished in %s seconds" % (time.time() - time_start))

MPI.Finalize
