import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from DataSet import *
import matplotlib.pyplot as plt
import matplotlib
import os

class OLSRegression:
    Regression = linear_model.LinearRegression()

    def __init__(self, DataFileName = 'TestData.csv', IndependentVariablesList = ['y'], DependentVariablesList = ['x', 'x2']):
        self.Data = DataSet(DataFileName)
        self.TrainingData = self.Data.TrainingData()
        self.TestingData = self.Data.TestingData()
        self.DependentVariablesList = DependentVariablesList
        self.IndependentVariablesList = IndependentVariablesList
        self.LowestDependentVariables = []
        self.AICs = []
        self.IteractionsNumber = 100

    def CalculateAICs(self):
        self.AICsCalculated = 0
        for IndependentVariable in self.IndependentVariablesList:
            self.IndependentVariable = IndependentVariable
            self.FindBestFit()
            print(self.LogMessage())
            self.PlotAICs()

    def FindBestFit(self):
        AICResetingFactor = 10e10
        self.LowestAIC = AICResetingFactor
        #self.IteractionsNumber = len(self.DependentVariablesList)
        #for i in range(self.IteractionsNumber):
        self.FindLowestAIC()

    def FindLowestAIC(self):
        #import random
        #random.shuffle(self.DependentVariablesList)
        self.BuildDataSet()
        #self.RemoveLowWeightVariables()

    def BuildDataSet(self):
        #i=1
        self.SortedVariables = self.SortedDependentVariables()

        print("sorted variables")
        print(self.SortedVariables)
        self.DependentVariables = []
        for var in self.SortedVariables:
            self.DependentVariables.append(var)
            self.SetUpLowestAIC()

    def SortedDependentVariables(self):
        self.DependentVariables = self.DependentVariablesList
        self.TrainTheModel()
        Sorted = [val for (_, val) in sorted(zip(self.Regression.coef_,
            self.DependentVariablesList), key=lambda x: x[0], reverse=True)]
        return Sorted

    def SetUpLowestAIC(self):
        self.TrainTheModel()
        if(self.AIC() < self.LowestAIC):
            self.LowestAIC = self.AIC()
            self.LowestDependentVariables = self.DependentVariables
            print(self.LowestAIC)
            print("Variables for LowestAIC")
            print(self.LowestDependentVariables)
        self.AICs.append(self.LowestAIC)
        #self.SaveResults()
        self.AICsCalculated+=1

    def RemoveLowWeightVariables(self):
        for coefficient in self.Regression.coef_:
            RemovalFactor = 0.001
            i = 0
            if (coefficient < RemovalFactor):
                self.DependentVariables.remove(self.DependentVariables[i])
                i+=1

    def PlotAICs(self):
        Index = range(len(self.AICs))
        Figure = plt.figure()
        SubImage = plt.subplot(111)
        print(self.AICs)
        SubImage.plot(Index, self.AICs, color='blue', label='$y = AICs' )
        FigureName = ' AICs plot '
        plt.title(FigureName)
        Figure.savefig( FigureName + '.png')

    def AIC(self):
        residuals = self.ModelFit() - self.TestingData[self.IndependentVariable]
        NonZeroValue = 0.000000000000000000001
        SumOfSquaredErrors = sum(np.power(residuals,2)) + NonZeroValue
        self.NumberOfVariables= len(self.DependentVariables) + 1
        self.NumberOfInstances = len(self.ModelFit())
        AIC = (self.NumberOfInstances * np.log(SumOfSquaredErrors /
            self.NumberOfInstances)) + (2 * self.NumberOfVariables)
        return AIC

    def AICc(self):
        return self.AIC() + (2 * np.power(self.NumberOfVariables,2) +
            (2 * self.NumberOfVariables)/ (self.NumberOfInstances -
            self.NumberOfVariables - 1))

    def TrainTheModel(self):
        self.Regression = linear_model.LinearRegression()
        self.Regression.fit(self.TrainingData[self.DependentVariables], self.TrainingData[self.IndependentVariable])

    def SaveResults(self):
        Index = self.Data.AllData().index.values
        Figure = plt.figure()
        SubImage = plt.subplot(111)
        SubImage.plot(Index, self.Data.AllData()[self.IndependentVariable], color='blue', label='$y = Data' )
        SubImage.plot(Index, self.Regression.predict(self.Data.AllData()[self.DependentVariables]), color='green', label='$y = Model')
        FigureName = self.IndependentVariable + ' AIC ' + str(self.LowestAIC)
        plt.title(FigureName)
        self.LogFolderName = 'Logs/' + self.IndependentVariable + '/'
        self.CreateLogFolder()
        Figure.savefig(self.LogFolderName + FigureName + '.png')

        self.WriteResults()

    def CreateLogFolder(self):
        if not os.path.exists(self.LogFolderName):
            os.makedirs(self.LogFolderName)

    def LogMessage(self):
        import datetime
        Log = '\n' + str(datetime.datetime.now())
        Log += '\nLowest AIC for: ' + str(self.IndependentVariable) + ' ' + str(self.AIC()) + '\nFound using: ' + str(self.LowestDependentVariables)
        Log +=  '\n' + str(self.Regression.intercept_) + str(self.Regression.coef_)
        Log += '\nMSE :' + str(mean_squared_error(self.TestingData[self.IndependentVariable], self.ModelFit()))
        Log += '\nAIC: ' + str(self.AIC())
        Log += '\nNumber of interactions: ' + str(self.AICsCalculated)
        return Log

    def WriteResults(self):
        import time
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.file = open(self.LogFolderName + 'AIC ' + str(self.LowestAIC) + timestr + '.log','a')
        self.file.write(self.LogMessage())
        self.file.close()

    def ModelFit(self):
        TestingData = self.TestingData[self.DependentVariables]
        ModelFit = self.Regression.predict(TestingData)
        return ModelFit

    def AICPlot(self):
        Index = self.Data.AllData().index.values
        Figure = plt.figure()
        SubImage = plt.subplot(111)
        SubImage.plot(Index, self.Data.AllData()[self.IndependentVariable], color='blue', label='$y = Data' )
        SubImage.plot(Index, self.Regression.predict(self.Data.AllData()[self.DependentVariables]), color='green', label='$y = Model')
        FigureName = self.IndependentVariable + ' AIC ' + str(self.LowestAIC)
        plt.title(FigureName)

        plt.show()
        Figure.savefig('Logs/' + FigureName + '.png')
