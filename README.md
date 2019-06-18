# load_forecasts_attack
Code repo for ACM e-Energy 2019 paper: Exploiting Vulnerabilities of Load Forecasting Through Adversarial Attacks

Extended version: Vulnerabilities of Power System Operations to Load Forecasting Data Injection Attacks

Authors: Yize Chen, Yushi Tan, Ling Zhang and Baosen Zhang

University of Washington

Contact: yizechen@uw.edu

## Introduction
Load forecasting plays a critical role in the operation and planning of power systems. By using input features such as historical loads and weather forecasts, system operators and utilities build forecast models to guide decision making in commitment and dispatch. As the forecasting techniques becomes more sophisticated, however, they also become more vulnerable to cybersecurity threats. We study the vulnerability of a class of load forecasting algorithms and analyze the potential impact on the power system operations, such as load shedding and increased dispatch costs. Specifically, we propose data injection attack algorithms that require minimal assumptions on the ability of the adversary. By only injecting malicious data in temperature from online weather forecast APIs, an attacker could manipulate load forecasts in arbitrary directions and cause significant and targeted damages to system operations. 

![alt text](https://github.com/chennnnnyize/load_forecasts_attack/blob/master/datasets/schematic.png)

## Attack Method
### Learn and Attack

### Gradient Estimation Attack

## Code Dependencies
All code are implemented in Python.

Learning algorithms for load forecasting: Tensorflow and Keras

Power systems unit commitment and economic dispatch: Pypsa



## Run Experiments with PreProcessed Datasets
We make use of weather data from Dark Sky API and load data from ENTSOE(European Network of Transmission System Operators for Electricity)

### June 17th: Updates on attacking a power network 

