from mpi4py import MPI

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
num_of_processes = comm.Get_size()

first_matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [3, 2, 1, 9],
    [4, 5, 6, 7],
]

second_matrix = [
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [1, 2, 3, 4],
]


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
    req = comm.isend(line, dest=0)
    req.wait()
else:
    result_matrix = [[] for i in range(0, num_of_processes)]
    result_matrix[0] = line_and_second_matrix_multiplication(
        first_matrix[0], second_matrix)
    for procid in range(1, num_of_processes):
        req = comm.irecv(source=procid)
        line = req.wait()
        result_matrix[line[0]] = line[1]
    print(result_matrix)

MPI.Finalize
