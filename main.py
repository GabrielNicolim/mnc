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
    print('\n Iterações: ', iterations)

    input()


def transposeMatrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    transposed = [[0] * rows for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]

    return transposed


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

    y = [0] * order

    for i in range(order):
        y[i] = vector[i]

        for j in range(i):
            y[i] -= L[i][j] * y[j]

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

    y = np.zeros(order)
    x = np.zeros(order)

    for i in range(order):
        y[i] = vector[i]

        for j in range(i):
            y[i] -= L[i][j] * y[j]

        y[i] /= L[i][i]

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
    x = np.array(initialGuess)

    newX = np.zeros(order)

    iterations = 0

    while iterations < maxIterations:
        for i in range(order):
            totalSum = 0

            for j in range(order):
                if i != j:
                    totalSum += matrix[i][j] * x[j]

            newX[i] = (vector[i] - totalSum) / matrix[i][i]

        if np.linalg.norm(newX - x) < tolerance:
            return newX, iterations + 1

        x = newX.copy()

        iterations += 1

    return newX, iterations


# Questão 09

def GaussSeidel(order, matrix, vector, initialGuess, tolerance, maxIterations):
    x = np.array(initialGuess)

    iterations = 0

    while iterations < maxIterations:
        newX = np.copy(x)

        for i in range(order):
            totalSum = 0

            for j in range(order):
                if i != j:
                    totalSum += matrix[i][j] * newX[j]

            newX[i] = (vector[i] - totalSum) / matrix[i][i]

        if np.linalg.norm(newX - x, np.inf) < tolerance:
            return newX, iterations + 1

        x = newX
        iterations += 1

    return newX, iterations


# Questão 10

def MatrizInversa(order, matrix):
    print("Escolha como deseja calcular")

    while True:
        print("\n1 - Método de decomposição LU")
        print("2 - Método de Gauss Compacto")

        type = int(input("\nEscolha sua opção: "))

        if (type < 1 or type > 2):
            continue

        if (type == 1):
        elif (type == 2):

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
        tolerance = getTolerance()
        maxIterations = getMaxIterations()

        vectorSolution, iterationsSolution = Jacobi(order, matrix, vector, tolerance, maxIterations)

        showVector(vectorSolution)
        showIterations(iterationsSolution)
    elif option == 9:
        order = getOrder()
        matrix = getMatrix(order)
        vector = getVector(order)
        tolerance = getTolerance()
        maxIterations = getMaxIterations()

        vectorSolution, iterationsSolution = GaussSeidel(order, matrix, vector, tolerance, maxIterations)

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
