import numpy as np


def getOrder():
    order = int(input("\nDigite a ordem da matriz: "))

    return order


def getMatrix(order):
    matrix = []

    print("\nDigite os elementos da matriz")

    for i in range(order):
        row = []

        for j in range(order):
            element = float(input(f"Digite o elemento da posição [{i + 1}][{j + 1}]: "))
            row.append(element)

        matrix.append(row)

    return matrix


def getVector(order):
    vector = []

    print("\nDigite os termos independentes")

    for i in range(order):
        element = float(input(f"Digite o termo independente da equação {i + 1}: "))

        vector.append(element)

    return vector


def getInitialGuess(order):
    vector = []

    print("\nDigite os termos da aproximação inicial para a solução")

    for i in range(order):
        element = float(input(f"Digite o termo da aproximação inicial {i + 1}: "))

        vector.append(element)

    return vector


def getTolerance():
    tolerance = float(input("\nDigite a precisão desejada: "))

    return tolerance


def getMaxIterations():
    maxIterations = int(input("\nDigite o número máximo de iterações: "))

    return maxIterations


def showDeterminat(determinant):
    print("\nDeterminante: ", determinant)

    input()


def showVector(vector):
    print("\nVetor: ", vector)

    input()


def showMatrix(matrix):
    print("\nMatrix: ", matrix)

    input()


def showIterations(iterations):
    print('\nIterações: ', iterations)

    input()


def transposeMatrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    transposed = [[0] * rows for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]

    return transposed

def getIdentity(order):
    identity = [[0 for _ in range(order)] for _ in range(order)]

    for i in range(order):
        identity[i][i] = 1

    return identity

# Questão 01
def CalculoDeterminante(order, matrix):
    if order == 1:
        return matrix[0][0]

    determinant = 0

    for j in range(order):
        minor = [row[:j] + row[j + 1:] for row in matrix[1:]]

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

    y = SistemaTriangularInferior(order, L, vector)

    return SistemaTriangularSuperior(order, U, y)


# Questão 05

def Cholesky(order, matrix, vector):
    L = np.zeros((order, order))

    for i in range(order):
        for j in range(i + 1):
            if i == j:
                L[i][j] = np.sqrt(matrix[i][i] - sum(L[i][k] ** 2 for k in range(j)))
            else:
                L[i][j] = (matrix[i][j] - sum(L[i][k] * L[j][k] for k in range(j))) / L[j][j]

    LT = transposeMatrix(L)

    y = SistemaTriangularInferior(order, L, vector)

    return SistemaTriangularSuperior(order, LT, y)


# Questão 06

def GaussCompacto(order, matrix, vector):
    for i in range(order):
        maxIndex = i
        maxValue = abs(matrix[i][i])

        for j in range(i + 1, order):
            if abs(matrix[j][i]) > maxValue:
                maxValue = abs(matrix[j][i])
                maxIndex = j

        matrix[i], matrix[maxIndex] = matrix[maxIndex], matrix[i]
        vector[i], vector[maxIndex] = vector[maxIndex], vector[i]

        for j in range(i + 1, order):
            factor = matrix[j][i] / matrix[i][i]
            vector[j] -= factor * vector[i]

            for k in range(i, order):
                matrix[j][k] -= factor * matrix[i][k]

    return SistemaTriangularSuperior(order, matrix, vector)


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

def Jacobi(order, matrix, vector, initialGuess, tolerance, maxIterations):
    new_guess = [0.0] * order

    iterations = 0

    while iterations < maxIterations:
        for i in range(order):
            total = 0

            for j in range(order):
                if i != j:
                    total += matrix[i][j] * initialGuess[j]

            new_guess[i] = (vector[i] - total) / matrix[i][i]

        norm = sum((new_guess[i] - initialGuess[i]) ** 2 for i in range(order)) ** 0.5

        if norm < tolerance:
            return new_guess, iterations + 1

        initialGuess = new_guess[:]

        iterations += 1

    return new_guess, iterations


# Questão 09

def GaussSeidel(order, matrix, vector, initialGuess, tolerance, maxIterations):
    for j in range(maxIterations):
        new_guess = initialGuess[:]

        for i in range(order):
            soma1 = sum(matrix[i][j] * new_guess[j] for j in range(i))

            soma2 = sum(matrix[i][j] * initialGuess[j] for j in range(i + 1, order))

            new_guess[i] = (vector[i] - soma1 - soma2) / matrix[i][i]

        if max(abs(new_guess[i] - initialGuess[i]) for i in range(order)) < tolerance:
            return new_guess, j

        initialGuess = new_guess[:]

    return initialGuess, maxIterations


# Questão 10

def MatrizInversa(order, matrix):
    print("Escolha como deseja calcular")

    while True:
        print("\n1 - Método de decomposição LU")
        print("2 - Método de Gauss Compacto")

        type = int(input("\nEscolha sua opção: "))

        if (type < 1 or type > 2):
            continue

        matrixSolution = np.zeros(order, order)

        if (type == 1):
            L = [[0.0] * order for _ in range(order)]
            U = [[0.0] * order for _ in range(order)]

            identity = getIdentity(order)

            for i in range(order):
                L[i][i] = 1.0

                for j in range(i, order):
                    total = sum(L[i][k] * U[k][j] for k in range(i))

                    U[i][j] = matrix[i][j] - total
                for j in range(i + 1, order):
                    total = sum(L[j][k] * U[k][i] for k in range(i))

                    L[j][i] = (matrix[j][i] - total) / U[i][i]

            reverse = []
            for col in range(order):
                y = SistemaTriangularInferior(order, L, [identity[row][col] for row in range(order)])

                x = SistemaTriangularSuperior(order, U, y)

                reverse.append(x)

            matrixSolution = transposeMatrix(reverse)
        elif (type == 2):
            identity = getIdentity(order)

            A = [row[:] for row in matrix]

            for i in range(order):
                A[i].extend(identity[i])

            for i in range(order):
                pivot = A[i][i]

                for j in range(2 * order):
                    A[i][j] /= pivot

                for k in range(order):
                    if k != i:
                        factor = A[k][i]
                        for j in range(2 * order):
                            A[k][j] -= factor * A[i][j]

            matrixSolution = [A[i][order:] for i in range(order)]

        return matrixSolution


while True:
    print("Escolha o que deseja calcular")
    print("1 - Calculo Determinante")
    print("2 - Sistema TriangularInferior")
    print("3 - Sistema TriangularSuperior")
    print("4 - Decomposicao LU")
    print("5 - Cholesky")
    print("6 - Gauss Compacto")
    print("7 - Gauss Jordan")
    print("8 - Jacobi")
    print("9 - Gauss Seidel")
    print("10 - Matriz Inversa")
    print("Outros - Encerra o programa")

    option = int(input("\nEscolha sua opção: "))

    if option == 1:
        order = getOrder()
        matrix = getMatrix(order)

        determinant = CalculoDeterminante(order, matrix)

        showDeterminat(determinant)
    elif option == 2:
        order = getOrder()
        matrix = getMatrix(order)
        vector = getVector(order)

        vectorSolution = SistemaTriangularInferior(order, matrix, vector)

        showVector(vectorSolution)
    elif option == 3:
        order = getOrder()
        matrix = getMatrix(order)
        vector = getVector(order)

        vectorSolution = SistemaTriangularSuperior(order, matrix, vector)

        showVector(vectorSolution)
    elif option == 4:
        order = getOrder()
        matrix = getMatrix(order)
        vector = getVector(order)

        vectorSolution = DecomposicaoLU(order, matrix, vector)

        showVector(vectorSolution)
    elif option == 5:
        order = getOrder()
        matrix = getMatrix(order)
        vector = getVector(order)

        vectorSolution = Cholesky(order, matrix, vector)

        showVector(vectorSolution)
    elif option == 6:
        order = getOrder()
        matrix = getMatrix(order)
        vector = getVector(order)

        vectorSolution = GaussCompacto(order, matrix, vector)

        showVector(vectorSolution)
    elif option == 7:
        order = getOrder()
        matrix = getMatrix(order)
        vector = getVector(order)

        vectorSolution = GaussJordan(order, matrix, vector)

        showVector(vectorSolution)
    elif option == 8:
        order = getOrder()
        matrix = getMatrix(order)
        vector = getVector(order)
        initialGuess = getInitialGuess(order)
        tolerance = getTolerance()
        maxIterations = getMaxIterations()

        vectorSolution, iterationsSolution = Jacobi(order, matrix, vector, initialGuess, tolerance, maxIterations)

        showVector(vectorSolution)
        showIterations(iterationsSolution)
    elif option == 9:
        order = getOrder()
        matrix = getMatrix(order)
        vector = getVector(order)
        initialGuess = getInitialGuess(order)
        tolerance = getTolerance()
        maxIterations = getMaxIterations()

        vectorSolution, iterationsSolution = GaussSeidel(order, matrix, vector, initialGuess, tolerance, maxIterations)

        showVector(vectorSolution)
        showIterations(iterationsSolution)
    elif option == 10:
        order = getOrder()
        matrix = getMatrix(order)

        matrixSolution = MatrizInversa(order, matrix)

        showVector(matrixSolution)
    else:
        print("\nPrograma encerrado")
        exit()
