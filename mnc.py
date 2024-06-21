import math

def interpolacaoDeNewton(n, points, x):
    xValues = [p[0] for p in points]
    yValues = [p[1] for p in points]

    coef = yValues[:]

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (xValues[i] - xValues[i - j])

    result = coef[0]
    product_term = 1.0

    for i in range(1, n):
        product_term *= (x - xValues[i - 1])
        result += coef[i] * product_term

    return result

def interpolacaoDeNewtonGregory(n, points, x):
    xValues = [p[0] for p in points]
    yValues = [p[1] for p in points]

    coef = yValues[:]

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (xValues[i] - xValues[i - j])

    h = xValues[1] - xValues[0]
    u = (x - xValues[0]) / h

    result = coef[0]
    productTerm = 1.0

    for i in range(1, n):
        productTerm *= (u - i + 1) / i
        result += coef[i] * productTerm

    return result

def ajusteLinear(n, points):
    sumX = sum(p[0] for p in points)
    sumY = sum(p[1] for p in points)
    sumXx = sum(p[0] * p[0] for p in points)
    sumXy = sum(p[0] * p[1] for p in points)

    a1 = (n * sumXy - sumX * sumY) / (n * sumXx - sumX * sumX)
    a0 = (sumY - a1 * sumX) / n

    yAdjusted = [a0 + a1 * p[0] for p in points]

    y_mean = sum(y for _, y in points) / n

    total = sum((y - y_mean) ** 2 for _, y in points)

    ss = sum((points[i][1] - yAdjusted[i]) ** 2 for i in range(n))

    rSquared = 1 - (ss / total)

    return a0, a1, yAdjusted, rSquared

def ajustePolinomial(n, degree, points):
    xPoints = [p[0] for p in points]
    yPoints = [p[1] for p in points]

<<<<<<< HEAD
    m = degree + 1
    A = [[sum(x ** (i + j) for x in xPoints) for j in range(m)] for i in range(m)]
    b = [sum(y * x ** i for y, x in zip(yPoints, xPoints)) for i in range(m)]
=======

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
            sum1 = sum(matrix[i][j] * new_guess[j] for j in range(i))

            sum2 = sum(matrix[i][j] * initialGuess[j] for j in range(i + 1, order))

            new_guess[i] = (vector[i] - sum1 - sum2) / matrix[i][i]

        if max(abs(new_guess[i] - initialGuess[i]) for i in range(order)) < tolerance:
            return new_guess, j

        initialGuess = new_guess[:]

    return initialGuess, maxIterations


# Questão 10

def MatrizInversa(order, matrix):
    print("Escolha como deseja calcular")
>>>>>>> b7b22f9014aa74e799b9607fab00b0bbf2861a61

    while True:
        print("Escolha o método de resolucao:")
        print("1. Decomposição LU")
        print("2. Decomposição Cholesky")
        choice = int(input("Escolha uma opção: "))

        if choice == 1:
            n = len(A)
            L = [[0.0] * n for _ in range(n)]
            U = [[0.0] * n for _ in range(n)]

            for i in range(n):
                L[i][i] = 1.0
                for j in range(i, n):
                    U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
                for j in range(i + 1, n):
                    L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

            n = len(b)
            y = [0] * n
            for i in range(n):
                y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))

            n = len(y)
            x = [0] * n
            for i in range(n - 1, -1, -1):
                x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]

            break
        elif choice == 2:
            n = len(A)
            L = [[0.0] * n for _ in range(n)]

            for i in range(n):
                for j in range(i + 1):
                    if i == j:
                        L[i][j] = (A[i][i] - sum(L[i][k] ** 2 for k in range(j))) ** 0.5
                    else:
                        L[i][j] = (A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))) / L[j][j]

            n = len(b)
            y = [0] * n
            for i in range(n):
                y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))

            Lt = [[L[j][i] for j in range(len(L))] for i in range(len(L))]

            n = len(y)
            x = [0] * n
            for i in range(n - 1, -1, -1):
                x[i] = (y[i] - sum(Lt[i][j] * x[j] for j in range(i + 1, n))) / Lt[i][i]

            break
        else:
            print("\nOpção inválida. Usando Decomposição LU por padrão.\n")

    yAdjusted = [sum(x[j] * x ** j for j in range(m)) for x in xPoints]

    y_mean = sum(y for _, y in points) / n

    total = sum((y - y_mean) ** 2 for _, y in points)

    ss = sum((points[i][1] - yAdjusted[i]) ** 2 for i in range(n))

    rSquared = 1 - (ss / total)

    return x, yAdjusted, rSquared


def ajusteExponencial(n, points):
    xPoints = [p[0] for p in points]
    yPoints = [math.log(p[1]) for p in points]

    sumX = sum(xPoints)
    sumY = sum(yPoints)
    sumXx = sum(x * x for x in xPoints)
    sumXy = sum(x * y for x, y in zip(xPoints, yPoints))

    b = (n * sumXy - sumX * sumY) / (n * sumXx - sumX * sumX)
    a = (sumY - b * sumX) / n
    a = math.exp(a)

    yAdjusted = [a * math.exp(b * x) for x in xPoints]

    y_mean = sum(y for _, y in points) / n

    total = sum((y - y_mean) ** 2 for _, y in points)

    ss = sum((points[i][1] - yAdjusted[i]) ** 2 for i in range(n))

    rSquared = 1 - (ss / total)

    return a, b, yAdjusted, rSquared

def getDesiredValue():
    desiredValue = float(input("\nDigite o valor de x: "))

    return desiredValue

while True:
    print("Escolha o que deseja calcular")
    print("1 - Interpolação Polinomial de Newton")
    print("2 - Interpolação Polinomial de Newton-Gregory")
    print("3 - Coeficiente de Determinação")
    print("4 - Ajuste de Reta")
    print("5 - Ajuste de Polinômio")
    print("6 - Ajuste de Curva Exponencial")
    print("Outros - Encerra o programa")

    option = int(input("\nEscolha sua opção: "))

    n = int(input("Quantos pontos deseja calcular: "))
    points = []

    for _ in range(n):
        x = float(input(f"Digite o valor de x [{ _ + 1 }]: "))
        y = float(input(f"Digite o valor de y [{ _ + 1 }]: "))

        points.append((x, y))

    if option == 1:
        x = getDesiredValue()

        print(f"\nO valor é aproximadamente: {interpolacaoDeNewton(n, points, x)}")

        input()
    elif option == 2:
        x = getDesiredValue()

        print(f"\nO valor é aproximadamente: {interpolacaoDeNewtonGregory(n, points, x)}")

        input()
    elif option == 3:
        yAdjusted = {}

        for i in range(n):
            yAdjusted[i] = float(input(f"Digite o valor para o ponto {i + 1}: "))

        y_mean = sum(y for _, y in points) / n

        total = sum((y - y_mean) ** 2 for _, y in points)

        ss = sum((points[i][1] - yAdjusted[i]) ** 2 for i in range(n))

        rSquared = 1 - (ss / total)

        print(f"\nO coeficiente é: {rSquared}")

        input()
    elif option == 4:
        a0, a1, yAdjusted, rSquared = ajusteLinear(n, points)

        print(f"\nEquação: y = {a0} + {a1}x")

        print(f"\nValores: {yAdjusted}")

        print(f"\nCoeficiente: {rSquared}")

        input()
    elif option == 5:
        degree = int(input("Digite o grau do polinômio: "))

        coef, yAdjusted, rSquared = ajustePolinomial(n, degree, points)

        print(f"\nCoeficientes: {coef}")

        print(f"\nValores: {yAdjusted}")

        print(f"\nCoeficiente: {rSquared}")
    elif option == 6:
        a, b, yAdjusted, rSquared = ajusteExponencial(n, points)

        print(f"\nEquação: y = {a}e^({b}x)")

        print(f"\nValores: {yAdjusted}")

        print(f"\nCoeficiente: {rSquared}")
    else:
        print("\nPrograma encerrado")
        exit()
