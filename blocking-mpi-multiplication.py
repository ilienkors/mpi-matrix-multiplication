from mpi4py import MPI
from random import randrange

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
num_of_processes = comm.Get_size()

first_matrix = [[randrange(10) for i in range(0, num_of_processes)] for j in range(0, num_of_processes)]
second_matrix = [[randrange(10) for i in range(0, num_of_processes)] for j in range(0, num_of_processes)]


def line_and_second_matrix_multiplication(line, second_matrix):
    matrix_size = len(line)
    result_line = []
    for i in range(matrix_size):
        line_sum = 0
        for j in range(matrix_size):
            line_sum += line[j] * second_matrix[j][i]
        result_line.append(line_sum)
    return result_line


if my_rank != 0:
    line = [my_rank, line_and_second_matrix_multiplication(
        first_matrix[my_rank], second_matrix)]
    comm.send(line, dest=0)
else:
    result_matrix = [[] for i in range(0, num_of_processes)]
    result_matrix[0] = line_and_second_matrix_multiplication(first_matrix[0], second_matrix)
    for procid in range(1, num_of_processes):
        line = comm.recv(source=procid)
        result_matrix[line[0]] = line[1]
    print(result_matrix)

MPI.Finalize
