import time
from mpi4py import MPI
from functions import line_and_second_matrix_multiplication
from functions import get_matrix

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
num_of_processes = comm.Get_size()

first_matrix = get_matrix(num_of_processes, 36.6)
second_matrix = get_matrix(num_of_processes, 1.23)

if my_rank == 0:
    time_start = time.time()

if my_rank != 0:
    line = [my_rank, line_and_second_matrix_multiplication(
        first_matrix[my_rank], second_matrix)]
    comm.send(line, dest=0)
else:
    result_matrix = [[] for i in range(0, num_of_processes)]
    result_matrix[0] = line_and_second_matrix_multiplication(
        first_matrix[0], second_matrix)
    for procid in range(1, num_of_processes):
        line = comm.recv(source=procid)
        result_matrix[line[0]] = line[1]
    # print(result_matrix)

if my_rank == 0:
    print("Finished in %s seconds" % (time.time() - time_start))

MPI.Finalize
