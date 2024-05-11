def calculateDeterminant(order, matrix):
    if len(matrix) != order or any(len(row) != order for row in matrix):
        raise ValueError("The matrix is not square")

    if order == 1:
        return matrix[0][0]

    determinant = 0
    for j in range(order):
        minor = [row[:j] + row[j+1:] for row in matrix[1:]]

        determinant += (-1) ** j * matrix[0][j] * calculateDeterminant(order - 1, minor)

    return determinant
