from mpi4py import MPI

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
num_of_processes = comm.Get_size()

second_matrix = [
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [1, 2, 3, 4],
]

if my_rank == 0:
    first_matrix = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [3, 2, 1, 9],
        [4, 5, 6, 7],
    ]
    first_matrix_row = [x for x in first_matrix]

else:
    first_matrix_row = None


def line_and_second_matrix_multiplication(line, second_matrix):
    matrix_size = len(line)
    result_line = []
    for i in range(matrix_size):
        line_sum = 0
        for j in range(matrix_size):
            line_sum += line[j] * second_matrix[j][i]
        result_line.append(line_sum)
    return result_line


first_matrix_row = comm.scatter(first_matrix_row, root=0)

line = [my_rank, line_and_second_matrix_multiplication(
    first_matrix_row, second_matrix)]

data = comm.gather(line, root=0)

if my_rank == 0:
    result_matrix = [[] for i in range(0, num_of_processes)]
    for row in data:
        result_matrix[row[0]] = row[1]

    print(result_matrix)

MPI.Finalize
