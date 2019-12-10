import pandas as pd
import numpy

class DataSet:
    def __init__(self, File, TestingRatio = 0.2):
        self.File = File
        self.TestingRatio = TestingRatio

    def AllData(self):
        data = pd.read_csv(self.File)
        data = data.dropna()
        return data

    def SampledData(self):
        data = self.AllData()
        SampledData = data.sample(frac=1)
        return SampledData

    def TrainingData(self):
        Data = self.SampledData()
        TestLength = numpy.round(len(Data.index) * self.TestingRatio)
        TestData = Data[: - int(TestLength)]
        return TestData

    def TestingData(self):
        Data = self.SampledData()
        TestLength = numpy.round(len(Data.index) * self.TestingRatio)
        TestData = Data[- int(TestLength) :]
        return TestData
