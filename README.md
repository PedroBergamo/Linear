<div align="center">
  <img src="./Linear/Resources/logo-square.png" width="200" height="200"/>
</div>

<div id="badges" align="center">
  <a href="https://www.linkedin.com/in/pedrobergamo/">
    <img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Badge"/>
  </a>
</div>

# Linear

Linear is a tool which allows you to find which combination of parameters gives the best fit in a linear regression according to AIC (Akaike Information Criterion).

## About

In summary, data scientists need their models to be precise but also to be simple, since complexity equals more computational power and less speed. AIC is a compromise between precision and complexity. The program takes a dataset with several parameters, and find the AIC for several singular and separate arrangements of them in order to discover the simplest model with higher precision.

## Example

Take the provided data set (TestData.csv) as an example. Imagine that you want to find the simplest best fit for 'z' using the other data sets ('y','x','x2','x3').

The software would  
the code finds the AIC for
procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

A straightforward application to solve linear equations using Scikit-learn linear regression models and .



## Bibliography
=======
In summary, data scientists need their models to be precise but also to be simple. Instead of MSE or r2 which measures only the precision of a model, AIC can be seen as a compromise between precision and complexity. Lower the AIC, the more "simply precise" is the model
In order to find the best regression model based on the lowest AIC of a certain dataset, the program plays around with different configurations of several parameters, and find the AIC for singular and separate arrangements of them.

## Installation

First, let's install all packages:
```
$ git clone git@github.com:PedroBergamo/Linear.git
$ cd ./Linear
$ python install.py
```

## Example

Take the provided data set (TestData.csv) as an example. Imagine the extreme scenario where you want to find the simplest best fit for 'x' using 'x' and other other data sets such as x^2 and x^3. If create a script and run:
```

$ from OLSRegression import *
$ OLSRegression('TestData.csv', ['x'], ['x2', 'x3','o','z','x']).CalculateAICs()
```
The software would find the AIC for a varied set of combinations of those dependent variables but it would eventually tell you that 'x' is the simplest best predictor for 'x'.


## Bibliography

- https://en.wikipedia.org/wiki/Linear_regression
- https://en.wikipedia.org/wiki/Akaike_information_criterion
