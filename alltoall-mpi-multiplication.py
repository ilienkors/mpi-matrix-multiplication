import time
from mpi4py import MPI
from numpy import array
from numpy import empty
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
    result_matrix = [[[] for j in range(0, num_of_processes)] for i in range(0, num_of_processes)]
    time_start = time.time()

first_matrix_row = comm.scatter(first_matrix_row, root=0)

senddata = array(line_and_second_matrix_multiplication(
    first_matrix_row, second_matrix))
recvdata = empty(num_of_processes, dtype=float)
comm.Alltoall(senddata, recvdata)

data = comm.gather(recvdata, root=0)

if my_rank == 0:
    for i, row in enumerate(data):
        for j in range(len(row)):
            result_matrix[j][i] = row[j]
    #print(result_matrix)

if my_rank == 0:
    print("Finished in %s seconds" % (time.time() - time_start))

MPI.Finalize

# mpirun -np 1 python3 alltoall-mpi-multiplication.py && mpirun -np 2 python3 alltoall-mpi-multiplication.py
# mpirun -np 4 python3 alltoall-mpi-multiplication.py && mpirun -np 8 python3 alltoall-mpi-multiplication.py
# mpirun -np 16 python3 alltoall-mpi-multiplication.py && mpirun -np 32 python3 alltoall-mpi-multiplication.py
# mpirun -np 64 python3 alltoall-mpi-multiplication.py && mpirun -np 128 python3 alltoall-mpi-multiplication.py
# mpirun -np 256 python3 alltoall-mpi-multiplication.py && mpirun -np 384 python3 alltoall-mpi-multiplication.py
# mpirun -np 512 python3 alltoall-mpi-multiplication.py

