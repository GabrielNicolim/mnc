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


