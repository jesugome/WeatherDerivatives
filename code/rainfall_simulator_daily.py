# Import needed libraries.
import os
import numpy as np
import pandas as pd
import datetime
from io import StringIO
import pickle
from dateutil.relativedelta import relativedelta
from scipy.stats import bernoulli
from scipy.stats import gamma
from scipy import stats
from joblib import Parallel, delayed
import time
import calendar

from ideam_data_reader import concatYearsPrecipitation
from enso_forescast_downloader import totalENSOforecastDownloader
import rainfall_simulator_glm_plus_vglm
import rainfall_simulator_glm_plus_gamma
import rainfall_simulator_glm_plus_vglm_noenso

# Simulates one run of simulation.
def oneRun_noenso(simulationDataFrame, amountParametersGamma):
    # Define the total rainfall amount over the simulation.
    rainfall = 0

    # Loop over days in simulation to calculate rainfall amount.
    for day in range(3, 4):

        todayState = 1

        # Computes total accumulated rainfall.
        if todayState == 1:

            # Additional data of day.
            todayProbNino = simulationDataFrame['probNino'][day]
            todayProbNina = simulationDataFrame['probNina'][day]
            todayMonth = simulationDataFrame['Month'][day]

            # Calculates gamma log(mu).
            gammaLogMu = amountParametersGamma['log_mu'][todayMonth]
            # Calculates gamma scale
            gammaLogShape = amountParametersGamma['log_shape'][todayMonth]

            # Update mu
            gammaMu = np.exp(gammaLogMu)

            # Update shape
            gammaShape = np.exp(gammaLogShape)

            # Calculate gamma scale.
            gammaScale = gammaMu / gammaShape

            # Generate random rainfall.
            todayRainfall = gamma.rvs(a=gammaShape, scale=gammaScale)

            # Updates rainfall amount.
            rainfall += todayRainfall
        else:
            # Write new day information.
            simulationDataFrame.loc[indexToday, 'Prep_in'] = 0

    return rainfall


# Run total iterations.
def totalRun_noenso(simulationDataFrame, amountParametersGamma, iterations):

    # Array to store all precipitations.
    rainfallPerIteration = [None] * iterations

    # Loop over each iteration(simulation)
    for i in range(iterations):
        simulationDataFrameC = simulationDataFrame.copy()
        iterationRainfall = oneRun_noenso(simulationDataFrameC, amountParametersGamma)

        rainfallPerIteration[i] = iterationRainfall

    # Recover start date of simulation.
    simulationStartDate = simulationDataFrame.index[1]

    return simulationStartDate, rainfallPerIteration


# Simulates one run of simulation.
def oneRun_enso(simulationDataFrame, amountParametersGamma):
    # Define the total rainfall amount over the simulation.
    rainfall = 0

    # Loop over days in simulation to calculate rainfall amount.
    for day in range(3, 4):

        todayState = 1

        # Computes total accumulated rainfall.
        if todayState == 1:

            # Additional data of day.
            todayProbNino = simulationDataFrame['probNino'][day]
            todayProbNina = simulationDataFrame['probNina'][day]
            todayMonth = simulationDataFrame['Month'][day]

            # Calculates gamma log(mu).
            gammaLogMu = amountParametersGamma['log_mu'][todayMonth] + todayProbNino*amountParametersGamma['log_mu'][13]
            # Calculates gamma scale
            gammaLogShape = amountParametersGamma['log_shape'][todayMonth]+todayProbNino * amountParametersGamma['log_shape'][13]

            # Update mu
            gammaMu = np.exp(gammaLogMu)

            # Update shape
            gammaShape = np.exp(gammaLogShape)

            # Calculate gamma scale.
            gammaScale = gammaMu / gammaShape

            # Generate random rainfall.
            todayRainfall = gamma.rvs(a=gammaShape, scale=gammaScale)

            # Updates rainfall amount.
            rainfall += todayRainfall
        else:
            # Write new day information.
            simulationDataFrame.loc[indexToday, 'Prep_in'] = 0

    return rainfall


# Run total iterations.
def totalRun_enso(simulationDataFrame, amountParametersGamma, iterations):

    # Array to store all precipitations.
    rainfallPerIteration = [None] * iterations

    # Loop over each iteration(simulation)
    for i in range(iterations):
        simulationDataFrameC = simulationDataFrame.copy()
        iterationRainfall = oneRun_enso(simulationDataFrameC, amountParametersGamma)

        rainfallPerIteration[i] = iterationRainfall

    # Recover start date of simulation.
    simulationStartDate = simulationDataFrame.index[1]

    return simulationStartDate, rainfallPerIteration


# Simulates one run of simulation.
def oneRun_enso_deltaIndex(simulationDataFrame, amountParametersGamma):
    # Define the total rainfall amount over the simulation.
    rainfall = 0

    # Loop over days in simulation to calculate rainfall amount.
    for day in range(3, 4):

        todayState = 1

        # Computes total accumulated rainfall.
        if todayState == 1:

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

            # Updates rainfall amount.
            rainfall += todayRainfall
        else:
            # Write new day information.
            simulationDataFrame.loc[indexToday, 'Prep_in'] = 0

    return rainfall


# Run total iterations.
def totalRun_enso_deltaIndex(simulationDataFrame, amountParametersGamma, iterations):

    # Array to store all precipitations.
    rainfallPerIteration = [None] * iterations

    # Loop over each iteration(simulation)
    for i in range(iterations):
        simulationDataFrameC = simulationDataFrame.copy()
        iterationRainfall = oneRun_enso_deltaIndex(simulationDataFrameC, amountParametersGamma)

        rainfallPerIteration[i] = iterationRainfall

    # Recover start date of simulation.
    simulationStartDate = simulationDataFrame.index[1]

    return simulationStartDate, rainfallPerIteration

def createTotalDataFrame(daysNumber, startDate, initialState, initialPrep, ensoForecast, optionMonthTerm):

    # Set variables names.
    totalDataframeColumns = ['state', 'Prep_in', 'Month', 'probNina', 'probNino', 'nextState']

    # Create dataframe.
    allDataDataframe = pd.DataFrame(columns=totalDataframeColumns)

    # Number of simulation days(i.e 30, 60)
    daysNumber = daysNumber

    # Simulation start date ('1995-04-22')
    startDate = startDate
    currentDate = datetime.datetime.strptime(startDate, '%Y-%m-%d')
    currentDate = currentDate + datetime.timedelta(days=-1)
    # State of rainfall last day before start date --> Remember 0 means dry and 1 means wet.
    initialState = initialState
    initialPrep = initialPrep  # Only fill when initialState == 1import calendar

    dates = pd.date_range(currentDate, periods=daysNumber + 2, freq='D')

    for date in dates:

        # Fill precipitation amount.
        allDataDataframe.loc[date.strftime('%Y-%m-%d'), 'Prep_in'] = np.nan

        # Fill month of date
        allDataDataframe.loc[date.strftime('%Y-%m-%d'), 'Month'] = date.month

        ensoDate = date + relativedelta(months=-optionMonthTerm)
        # Fill El Nino ENSO forecast probability.
        allDataDataframe.loc[date.strftime('%Y-%m-%d'), 'probNino'] = float(
            ensoForecast[ensoDate.strftime('%Y-%m')].loc[optionMonthTerm - 1, 'El Niño'].strip('%').strip('~')) / 100

        # Fill La Nina ENSO forecast probability.
        allDataDataframe.loc[date.strftime('%Y-%m-%d'), 'probNina'] = float(
            ensoForecast[ensoDate.strftime('%Y-%m')].loc[optionMonthTerm - 1, 'La Niña'].strip('%').strip('~')) / 100

        # Fill State.
        allDataDataframe.loc[date.strftime('%Y-%m-%d'), 'state'] = np.nan

    simulationDataFrame = allDataDataframe[:-1]

    # Add column with  with delta_Index = pNiño - pNiña
    simulationDataFrame = simulationDataFrame.assign(deltaIndex=simulationDataFrame.probNino - simulationDataFrame.probNina)

    # Fill initial conditions.
    simulationDataFrame['state'][0] = initialState
    if initialState == 1:
        simulationDataFrame['Prep_in'][0] = initialPrep
    else:
        simulationDataFrame['Prep_in'][0] = 0.0

    return simulationDataFrame


def run_simulation_per_date(startDate,model):

    ## Generates initial conditions.
    # Defines initial state based on proportions.
    successProbability = 0.5
    initialState = bernoulli.rvs(successProbability)
    # Calculates initial precipitation.
    if initialState == 1:
        initialPrep = 0.1
    else:
        initialPrep = 0.0

    optionMonthTerm = 1

    # Define number of days to simulated based on month.
    currentDate = datetime.datetime.strptime(startDate, '%Y-%m-%d')
    currentMonth = currentDate.month
    currentYear = currentDate.year
    days_number = calendar.monthrange(currentYear, currentMonth)[1]


    # Load enso forecast.
    ensoForecast = pickle.load(open('../datasets/ensoForecastProb/ensoForecastProbabilities.pickle', 'rb'))

    ## Create dataframe to simulate.
    simulationDataFrame = createTotalDataFrame(daysNumber=days_number, startDate=startDate, initialState=initialState,
                                               initialPrep=initialPrep, ensoForecast=ensoForecast,
                                               optionMonthTerm=optionMonthTerm)
    if model == 'basic':
        pass
    elif model == 'glm_gamma':
        simulationStartDate, rainfallPerIteration, wetDaysPerIteration = rainfall_simulator_glm_plus_gamma.totalRun(
            simulationDataFrame=simulationDataFrame,
            transitionsParametersDry=transitionsParametersDry,
            transitionsParametersWet=transitionsParametersWet, amountParametersGamma=fittedGamma,
            iterations=1000)
        pass

    elif model == 'glm_vglm_noenso':

        ### Load transitions and amount parameters.

        amountParametersGamma = pd.read_csv('../results/visibleMarkov/amountGamma_vglm_noenso.csv', sep=' ', header=None,
                                            names=['variable', 'log_mu', 'log_shape'])
        amountParametersGamma.index += 1

        # Run simulator.
        simulationStartDate, rainfallPerIteration = totalRun_noenso(simulationDataFrame=simulationDataFrame,
                                                                    amountParametersGamma=amountParametersGamma,
                                                                    iterations=1000)

    elif model == 'glm_vglm':
        ### Load transitions and amount parameters.

        amountParametersGamma = pd.read_csv('../results/visibleMarkov/amountGamma_vglm.csv', sep=' ', header=None,
                                            names=['variable', 'log_mu', 'log_shape'])
        amountParametersGamma.index += 1

        # Run simulator.
        simulationStartDate, rainfallPerIteration = totalRun_enso(simulationDataFrame=simulationDataFrame,
                                                                  amountParametersGamma=amountParametersGamma,
                                                                  iterations=1000)
    elif model == 'glm_vglm_deltaIndex':
        ### Load transitions and amount parameters.

        amountParametersGamma = pd.read_csv('../results/visibleMarkov/amountGamma_vglm_deltaIndex.csv', sep=' ', header=None,
                                            names=['variable', 'log_mu', 'log_shape'])
        amountParametersGamma.index += 1

        # Run simulator.
        simulationStartDate, rainfallPerIteration = totalRun_enso_deltaIndex(simulationDataFrame=simulationDataFrame,
                                                                             amountParametersGamma=amountParametersGamma,
                                                                             iterations=1000)
    else:
        print('Please choose a valid model to simulate:\n\t-"basic"\n\t-"glm_gamma"\n\t-"glm_vgam"')

    print('Simulation for: ', simulationStartDate, ' is done.')

    return simulationStartDate, rainfallPerIteration


if __name__ == "__main__":

    # Loop to iterate over dates and simulates raining.
    model = 'glm_vglm_deltaIndex'  # 'Please choose a valid model to simulate: -"basic" -"glm_gamma" -"glm_vgam" -"glm_gamma_noenso" -"glm_vgam_noenso" -"glm_vgam_deltaIndex"'

    # Load all possible dates to simulate.
    startDate = '2005-01-01'
    endDate = '2015-12-31'

    dates = pd.date_range(start=startDate, end = endDate, freq='MS')
    dates = list(dates.strftime('%Y-%m-%d'))

    # Initialize time
    startTime = time.time()

    simulations = Parallel(n_jobs=-1, verbose=10)(delayed(run_simulation_per_date)(date,model) for date in dates)

    # Calculate time
    currentTime = time.time() - startTime
    with open('../results/simulations/simulation_daily_'+model+'.pickle', 'wb') as f:
        pickle.dump(simulations, f)

    print('Simulations for all dates have finished successfully.')
    # Logging time.
    print('The elapsed time over simulation is: ', currentTime, ' seconds.')

    # Comment to know how to open file.
    '''
    with open('../results/simulations/simulation_daily_'+model+'.pickle', 'rb') as f:
        data = pickle.load(f)
    '''

    ### Export to excel.

    # Open simulations.
    with open('../results/simulations/simulation_daily_'+model+'.pickle', 'rb') as f:
        simulation = pickle.load(f)

    iterations = ['iteration_'+str(i) for i in range(1, 1001)]

    precipitation = pd.DataFrame(columns=iterations)
    wetDays = pd.DataFrame(columns=iterations)
    for date in simulation:
        precipitation.loc[date[0]] = date[1]

    # Export dataframes to excel.
    precipitation.to_excel('../results/simulations/precipitation_daily_simulation_'+model+'.xlsx')



