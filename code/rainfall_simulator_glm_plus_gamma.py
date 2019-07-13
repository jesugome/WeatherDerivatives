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
from scipy import stats
import os


def fit_gamma():

    if not os.path.exists('../results/visibleMarkov/fittedGamma.csv'):
        # Same dates of vglm.
        startDate = '2005-01-01'
        endDate = '2015-12-31'

        # Read precipitation over time.
        precipitationAllTime = pd.read_csv('../datasets/fullDataset/completeDailyDataset_' + startDate + '_' + endDate + '.csv')
        noZeroPrecipitation = precipitationAllTime[precipitationAllTime['Prep_in'] != 0.0]

        gammaAlpha, gammaLoc, gammaBeta = stats.gamma.fit(noZeroPrecipitation['Prep_in'])

        fittedParameter = {
            'Shape': [gammaAlpha],
            'Loc': [gammaLoc],
            'Scale': [gammaBeta]
        }

        fittedGamma = pd.DataFrame(fittedParameter)
        fittedGamma.to_csv('../results/visibleMarkov/fittedGamma.csv')
    return True

# Updates the state of the day based on yesterday state.
def updateState(yesterdayIndex, simulationDataFrame, transitionsParametersDry, transitionsParametersWet):
    # Additional data of day.
    yesterdayState = simulationDataFrame['state'][yesterdayIndex]
    yesterdayPrep = simulationDataFrame['Prep_in'][yesterdayIndex]
    yesterdayProbNino = simulationDataFrame['probNino'][yesterdayIndex]
    yesterdayProbNina = simulationDataFrame['probNina'][yesterdayIndex]
    yesterdayMonth = simulationDataFrame['Month'][yesterdayIndex]

    # Calculate transition probability.
    if yesterdayState == 0:
        # Includes month factor + probNino value + probNino value.
        successProbabilityLogit = transitionsParametersDry['value'][yesterdayMonth] \
                                  + yesterdayProbNino * transitionsParametersDry['value'][13]

        successProbability = (np.exp(successProbabilityLogit)) / (1 + np.exp(successProbabilityLogit))

    elif yesterdayState == 1:
        # Includes month factor + probNino value + probNino value + prep value .
        successProbabilityLogit = transitionsParametersWet['value'][yesterdayMonth] \
                                  + yesterdayProbNino * transitionsParametersWet['value'][13]

        successProbability = (np.exp(successProbabilityLogit)) / (1 + np.exp(successProbabilityLogit))
    else:
        print('State of date: ', simulationDataFrame.index[yesterdayIndex], ' not found.')

    todayState = bernoulli.rvs(successProbability)

    return todayState


# Simulates one run of simulation.
def oneRun(simulationDataFrame, transitionsParametersDry, transitionsParametersWet, fittedGamma):
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

            # Generate today rainfall.
            todayRainfall = gamma.rvs(fittedGamma['Shape'][0], fittedGamma['Loc'][0], fittedGamma['Scale'][0])

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

