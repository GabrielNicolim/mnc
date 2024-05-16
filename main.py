import numpy as np

# Questão 01
def CalculoDeterminante(order, matrix):
    if order == 1:
        return matrix[0][0]

    determinant = 0
    for j in range(order):
        minor = [row[:j] + row[j+1:] for row in matrix[1:]]

        determinant += (-1) ** j * matrix[0][j] * CalculoDeterminante(order - 1, minor)

    return determinant

# Questão 02

def SistemaTriangularInferior(order, matrix, vector):
    solution = [0] * order

    for i in range(order):
        solution[i] = vector[i]

        for j in range(i):
            solution[i] -= matrix[i][j] * solution[j]

        solution[i] /= matrix[i][i]

    return solution

# Questão 03

def SistemaTriangularSuperior(order, matrix, vector):
    solution = [0] * order

    for i in range(order - 1, -1, -1):
        solution[i] = vector[i]

        for j in range(i + 1, order):
            solution[i] -= matrix[i][j] * solution[j]

        solution[i] /= matrix[i][i]

    return solution

# Questão 04

def DecomposicaoLU(order, matrix, vector):
    L = [[0] * order for _ in range(order)]
    U = [[0] * order for _ in range(order)]

    for i in range(order):
        for k in range(i, order):
            sum = 0

            for j in range(i):
                sum += (L[i][j] * U[j][k])

            U[i][k] = matrix[i][k] - sum

        for k in range(i, order):
            if i == k:
                L[i][i] = 1
            else:
                sum = 0
                for j in range(i):
                    sum += (L[k][j] * U[j][i])

                L[k][i] = (matrix[k][i] - sum) / U[i][i]

    y = [0] * order
    x = [0] * order

    for i in range(order):
        y[i] = vector[i]

        for j in range(i):
            y[i] -= L[i][j] * y[j]

    for i in range(order - 1, -1, -1):
        x[i] = y[i]

        for j in range(i + 1, order):
            x[i] -= U[i][j] * x[j]

        x[i] /= U[i][i]

    return x

# Questão 05

def transposeMatrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    transposed = [[0] * rows for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]

    return transposed

def Cholesky(order, matrix, vector):
    L = np.zeros((order, order))

    for i in range(order):
        for j in range(i + 1):
            if i == j:
                L[i][j] = np.sqrt(matrix[i][i] - sum(L[i][k] ** 2 for k in range(j)))
            else:
                L[i][j] = (matrix[i][j] - sum(L[i][k] * L[j][k] for k in range(j))) / L[j][j]

    LT = transposeMatrix(L)

    y = np.zeros(order)
    x = np.zeros(order)

    for i in range(order):
        y[i] = vector[i]

        for j in range(i):
            y[i] -= L[i][j] * y[j]

        y[i] /= L[i][i]

    for i in range(order - 1, -1, -1):
        x[i] = y[i]

        for j in range(i + 1, order):
            x[i] -= LT[i][j] * x[j]

        x[i] /= LT[i][i]

    return x

# Questão 06

def GaussCompact(order, matrix, vector):
    for i in range(order):
        max_index = i
        max_value = abs(matrix[i][i])

        for j in range(i + 1, order):
            if abs(matrix[j][i]) > max_value:
                max_value = abs(matrix[j][i])
                max_index = j

        matrix[i], matrix[max_index] = matrix[max_index], matrix[i]
        vector[i], vector[max_index] = vector[max_index], vector[i]

        for j in range(i + 1, order):
            factor = matrix[j][i] / matrix[i][i]
            vector[j] -= factor * vector[i]

            for k in range(i, order):
                matrix[j][k] -= factor * matrix[i][k]

    solution = [0] * order
    for i in range(order - 1, -1, -1):
        solution[i] = vector[i]

        for j in range(i + 1, order):
            solution[i] -= matrix[i][j] * solution[j]

        solution[i] /= matrix[i][i]

    return solution

# Questão 07

def GaussJordan(order, matrix, vector):
    for i in range(order):
        pivot = matrix[i][i]

        for j in range(order):
            matrix[i][j] /= pivot

        vector[i] /= pivot

        for j in range(order):
            if i != j:
                factor = matrix[j][i]

                for k in range(order):
                    matrix[j][k] -= factor * matrix[i][k]

                vector[j] -= factor * vector[i]

    return vector

# Questão 08

def Jacobi(order, matrix, vector, initial_guess, tolerance, max_iterations):
    x = np.array(initial_guess)

    x_new = np.zeros(order)

    iterations = 0

    while iterations < max_iterations:
        for i in range(order):
            sum_ = 0

            for j in range(order):
                if i != j:
                    sum_ += matrix[i][j] * x[j]

            x_new[i] = (vector[i] - sum_) / matrix[i][i]

        if np.linalg.norm(x_new - x) < tolerance:
            return x_new, iterations + 1

        x = x_new.copy()

        iterations += 1

    return x_new, iterations

# Questão 09


