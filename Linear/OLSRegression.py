import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from DataSet import *
import matplotlib.pyplot as plt
import matplotlib
import os

class OLSRegression:
    Regression = linear_model.LinearRegression()

    def __init__(self, DataFileName, IndependentVariablesList, DependentVariablesList):
        self.Data = DataSet(DataFileName)
        self.TrainingData = self.Data.TrainingData()
        self.TestingData = self.Data.TestingData()
        self.DependentVariablesList = DependentVariablesList
        self.IndependentVariablesList = IndependentVariablesList
        self.CalculateAICs()

    def CalculateAICs(self):
        for IndependentVariable in self.IndependentVariablesList:
            self.IndependentVariable = IndependentVariable
            self.DoTheRegression()
            print(self.LogMessage())

    def DoTheRegression(self):
        AICResetingFactor = 10e10
        self.LowestAIC = AICResetingFactor
        IteractionsNumber = len(self.DependentVariablesList)
        for i in range(IteractionsNumber):
            self.FindLowestAICs()

    def FindLowestAICs(self):
        import random
        random.shuffle(self.DependentVariablesList)
        self.DependentVariables = []
        for Variable in self.DependentVariablesList:
            self.DependentVariables.append(Variable)
            self.SetLowestAIC()
        self.SaveResults()

    def SetLowestAIC(self):
        self.TrainTheModel()
        if(self.AIC() < self.LowestAIC):
            self.LowestAIC = self.AIC()

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
        Log += '\nLowest AIC for: ' + str(self.IndependentVariable) +' '+ str(self.LowestAIC)+ '\nFound using: ' + str(self.DependentVariables)
        Log +=  '\n' + str(self.Regression.intercept_) + str(self.Regression.coef_)
        Log += '\nMSE :' + str(mean_squared_error(self.TestingData[self.IndependentVariable], self.ModelFit()))
        Log += '\nAIC: ' + str(self.AIC())
        return Log

    def AIC(self):
        import numpy as np
        residuals = self.ModelFit() - self.TestingData[self.IndependentVariable]
        NonZeroValue = 0.000000000000000000001
        SumOfSquaredErrors = sum(np.power(residuals,2)) + NonZeroValue
        NumberOfVariables= len(self.DependentVariables) + 1
        NumberOfInstances = len(self.ModelFit())
        AIC = (NumberOfInstances * np.log(SumOfSquaredErrors / NumberOfInstances)) + (2 * NumberOfVariables)
        return AIC

    def WriteResults(self):
        import time
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.file = open(self.LogFolderName + 'AIC' + str(self.LowestAIC) + timestr + '.log','a')
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
