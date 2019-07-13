# Import needed libraries.
import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from scipy.stats import gamma
import pickle
import time
import datetime

# Updates the state of the day based on yesterday state.
def updateState(yesterdayIndex, simulationDataFrame, transitionsParametersDry, transitionsParametersWet):
    # Additional data of day.
    yesterdayState = simulationDataFrame['state'][yesterdayIndex]
    yesterdayPrep = simulationDataFrame['Prep_in'][yesterdayIndex]
    yesterdayProbNino = simulationDataFrame['probNino'][yesterdayIndex]
    yesterdayProbNina = simulationDataFrame['probNina'][yesterdayIndex]
    yesterdayDeltaIndex = simulationDataFrame['deltaIndex'][yesterdayIndex]
    yesterdayMonth = simulationDataFrame['Month'][yesterdayIndex]

    # Calculate transition probability.
    if yesterdayState == 0:
        # Includes month factor + probNino value + probNino value.
        successProbabilityLogit = transitionsParametersDry['value'][yesterdayMonth] \
                                  + yesterdayDeltaIndex * transitionsParametersDry['value'][13]

        successProbability = (np.exp(successProbabilityLogit)) / (1 + np.exp(successProbabilityLogit))

    elif yesterdayState == 1:
        # Includes month factor + probNino value + probNino value + prep value .
        successProbabilityLogit = transitionsParametersWet['value'][yesterdayMonth] \
                                  + yesterdayDeltaIndex * transitionsParametersWet['value'][13]

        successProbability = (np.exp(successProbabilityLogit)) / (1 + np.exp(successProbabilityLogit))
    else:
        print('State of date: ', simulationDataFrame.index[yesterdayIndex], ' not found.')

    todayState = bernoulli.rvs(successProbability)

    return todayState


# Simulates one run of simulation.
def oneRun(simulationDataFrame, transitionsParametersDry, transitionsParametersWet, amountParametersGamma):
    # Define the total rainfall amount over the simulation.
    rainfall = 0

    # Total rainfall days.
    wetDays = 0

    # Loop over days in simulation to calculate rainfall amount.
    for day in range(1, len(simulationDataFrame)):

        # Get today date.
        dateOfDay = datetime.datetime.strptime(simulationDataFrame.index[day], '%Y-%m-%d')
        indexToday = simulationDataFrame.index[day]
        # Get today date.
        dateOfYesterday = datetime.datetime.strptime(simulationDataFrame.index[day - 1], '%Y-%m-%d')
        indexYesterday = simulationDataFrame.index[day - 1]

        # Update today state based on the yesterday state.
        todayState = updateState(day - 1, simulationDataFrame, transitionsParametersDry, transitionsParametersWet)

        # Write new day information.
        simulationDataFrame.loc[indexToday, 'state'] = todayState
        simulationDataFrame.loc[indexYesterday, 'nextState'] = todayState

        # Computes total accumulated rainfall.
        if todayState == 1:

            # Sum wet day.
            wetDays += 1

            # Additional data of day.
            todayProbNino = simulationDataFrame['probNino'][day]
            todayProbNina = simulationDataFrame['probNina'][day]
            todayDeltaIndex = simulationDataFrame['deltaIndex'][day]
            todayMonth = simulationDataFrame['Month'][day]

            # Calculates gamma log(mu).
            gammaLogMu = amountParametersGamma['log_mu'][todayMonth] + todayDeltaIndex*amountParametersGamma['log_mu'][13]
            # Calculates gamma scale
            gammaLogShape = amountParametersGamma['log_shape'][todayMonth] + todayDeltaIndex*amountParametersGamma['log_shape'][13]

            # Update mu
            gammaMu = np.exp(gammaLogMu)

            # Update shape
            gammaShape = np.exp(gammaLogShape)

            # Calculate gamma scale.
            gammaScale = gammaMu / gammaShape

            # Generate random rainfall.
            todayRainfall = gamma.rvs(a=gammaShape, scale=gammaScale)

            # Write new day information.
            simulationDataFrame.loc[indexToday, 'Prep_in'] = todayRainfall

            # Updates rainfall amount.
            rainfall += todayRainfall
        else:
            # Write new day information.
            simulationDataFrame.loc[indexToday, 'Prep_in'] = 0

        yesterdayState = todayState

    return rainfall, wetDays


# Run total iterations.
def totalRun(simulationDataFrame, transitionsParametersDry, transitionsParametersWet, amountParametersGamma,
               iterations):
    # Initialize time
    startTime = time.time()

    # Array to store all precipitations.
    rainfallPerIteration = [None] * iterations
    wetDaysPerIteration = [None] * iterations

    # Loop over each iteration(simulation)
    for i in range(iterations):
        simulationDataFrameC = simulationDataFrame.copy()
        iterationRainfall, wetDays = oneRun(simulationDataFrameC, transitionsParametersDry, transitionsParametersWet, amountParametersGamma)

        rainfallPerIteration[i] = iterationRainfall
        wetDaysPerIteration[i] = wetDays

    # Calculate time
    currentTime = time.time() - startTime

    # Print mean of wet days.

    # print('The mean of wet days is: ', np.mean(wetDaysPerIteration))

    # Logging time.
    # print('The elapsed time over simulation is: ', currentTime, ' seconds.')

    # Recover start date of simulation.
    simulationStartDate = simulationDataFrame.index[1]

    return simulationStartDate, rainfallPerIteration, wetDaysPerIteration

