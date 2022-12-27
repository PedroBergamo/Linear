from OLSRegression import *

clear = lambda: os.system('cls')
OLSRegression('TestData.csv', ['x'], ['x2', 'x3','o','z','x']).CalculateAICs()
