def line_and_second_matrix_multiplication(line, second_matrix):
    matrix_size = len(line)
    result_line = []
    for i in range(matrix_size):
        line_sum = 0
        for j in range(matrix_size):
            line_sum += line[j] * second_matrix[j][i]
        result_line.append(line_sum)
    return result_line

def get_matrix(size, number):
    return [[number for j in range(size)] for i in range(size)]    