import unittest
from DataSet import *
from pandas import *
from OLSRegression import *
import numpy
import pytest
import os

class OLSUnitTests(unittest.TestCase):
    Data = DataSet('TestData.csv')
    OLSTest = OLSRegression('TestData.csv', 'y', ['x', 'x2'])

    def test_FitSize(self):
        NewOLS = OLSRegression('TestData.csv', 'x', ['x'])
        Actual = len(NewOLS.ModelFit())
        Expected = len(NewOLS.Data.TestingData()['x'])
        self.assertEqual(Actual, Expected)

    def test_DataSetWithParameters(self):
        TrainingDataLenght = len(self.Data.TrainingData())
        Expected = numpy.round(len(self.Data.AllData().index) * 0.8)
        self.assertEqual(TrainingDataLenght, Expected)

    def test_TestingSet(self):
        Actual = len(self.Data.TestingData())
        Expected = numpy.round(len(self.Data.AllData().index) * 0.2)
        self.assertEqual(Actual, Expected)

    def test_SetDependentVariableList(self):
        Actual = self.OLSTest.DependentVariablesList
        Expected = ['x', 'x2']
        self.assertEqual(Actual, Expected)

    def test_RegressionCoefficient(self):
        NewOLS = OLSRegression('TestData.csv', 'y', ['x'])
        Actual = numpy.round(NewOLS.Regression.coef_)
        self.assertEqual(Actual, 2)

    def test_AIC(self):
        NewOLS = OLSRegression('TestData.csv', 'x', ['x'])
        Actual = numpy.round(NewOLS.AIC())
        self.assertEqual(Actual, -94)

if __name__ == '__main__':
    clear = lambda: os.system('cls')
    clear()

    unittest.main()
