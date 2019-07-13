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
import rainfall_simulator_glm_plus_vglm_deltaIndex


def total_dataset_constructor(startDate, endDate):

    # Configure path to read datasets.

    path = '../datasets/'

    # Download the update dataset for precipitation.
    if not os.path.exists(path + 'precipitationAllTime.csv'):
        command = 'wget https://github.com/jesugome/WeatherDerivates/raw/master/datasets/precipitationAllTime.csv -P ' + path
        os.system(command)

    precipitationAllTime = pd.read_csv(path + 'precipitationAllTime.csv', header=None, names=['Date', 'Prep'])
    precipitationAllTime['Date'] = pd.to_datetime(precipitationAllTime['Date'])
    precipitationAllTime = precipitationAllTime.set_index('Date')

    print('* Note: Precipitations are in mmm(milimetros de agua)--- 1mm = 1L/m²')

    # Download the update dataset for indices.
    if not os.path.exists(path + 'nino34_daily.dat'):
        command = 'wget https://github.com/jesugome/WeatherDerivates/raw/master/datasets/nino34_daily.dat -P ' + path
        os.system(command)

    nino34AllTime = pd.read_csv(
        StringIO(''.join(l.replace('   ', '  ').replace('  ', ' ') for l in open(path + 'nino34_daily.dat'))),
        header=None, names=['Date', 'nino34'], skiprows=lambda x: x in range(0, 100), sep=' ')
    nino34AllTime['Date'] = nino34AllTime['Date'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d'))
    nino34AllTime = nino34AllTime.set_index('Date')

    # Load ENSO probabilistic forecast.
    ensoForecast = pickle.load(open('../../datasets/ensoForecastProb/ensoForecastProbabilities.pickle', 'rb'))

    # Create dataframe with all information..
    totalDataframeColumns = ['Prep_mm', 'Month', 'nino34', 'probNeutral', 'probNino', 'probNina', 'state', 'nextState']
    allDataDataframe = pd.DataFrame(columns=totalDataframeColumns)

    # Fill datafame information.

    # Generate a timestamp with all days in simulation.
    dates = pd.date_range(startDate, endDate, freq='D')

    for date in dates:

        # Fill precipitation amount.
        allDataDataframe.loc[date.strftime('%Y-%m-%d'), 'Prep_mm'] = precipitationAllTime.loc[
            date.strftime('%Y-%m-%d'), 'Prep']

        # Fill month of date
        allDataDataframe.loc[date.strftime('%Y-%m-%d'), 'Month'] = date.month

        # Fill daily nino 34.
        allDataDataframe.loc[date.strftime('%Y-%m-%d'), 'nino34'] = nino34AllTime.loc[
            date.strftime('%Y-%m-%d'), 'nino34']

        # ENSO probabilities is given by the forecast of the last month.
        #ensoDate = date + relativedelta(months=-1)
        ensoDate = date
        # Fill neutral Enso forecast probability.
        allDataDataframe.loc[date.strftime('%Y-%m-%d'), 'probNeutral'] = float(ensoForecast[ensoDate.strftime('%Y-%m')].loc[0, 'Neutral'].strip('%').strip('~')) / 100

        # Fill El Nino ENSO forecast probability.
        allDataDataframe.loc[date.strftime('%Y-%m-%d'), 'probNino'] = float(ensoForecast[ensoDate.strftime('%Y-%m')].loc[0, 'El Niño'].strip('%').strip('~')) / 100

        # Fill La Nina ENSO forecast probability.
        allDataDataframe.loc[date.strftime('%Y-%m-%d'), 'probNina'] = float(ensoForecast[ensoDate.strftime('%Y-%m')].loc[0, 'La Niña'].strip('%').strip('~')) / 100

        # Fill State.
        allDataDataframe.loc[date.strftime('%Y-%m-%d'), 'state'] = (1 if precipitationAllTime.loc[date.strftime('%Y-%m-%d'), 'Prep'] > 0 else 0)

        # Fill next State.
        nextDate = date + 1

        if date != dates[-1]:
            allDataDataframe.loc[date.strftime('%Y-%m-%d'), 'nextState'] = (
                1 if precipitationAllTime.loc[nextDate.strftime('%Y-%m-%d'), 'Prep'] > 0 else 0)

    allDataDataframe = allDataDataframe[:-1]

    # Add columm with precipitation in inches.
    allDataDataframe = allDataDataframe.assign(Prep_in=allDataDataframe.Prep_mm/25.4)

    # Add column with delta_Index = pNiño - pNiña
    allDataDataframe.insert(loc=6, column='deltaIndex', value=pd.Series(allDataDataframe.probNino - allDataDataframe.probNina))
    # Save total data to .csv
    if not os.path.exists('../datasets/fullDataset/'):
        os.mkdir('../datasets/fullDataset/')
    allDataDataframe.to_csv('../datasets/fullDataset/completeDailyDataset_' + startDate + '_' + endDate + '.csv')

    return allDataDataframe


def recover_Transitions_parameters_from_R():

    # Load transitions and amount parameters.

    # Transitions probabilites.
    transitionsParametersDry = pd.read_csv('../results/visibleMarkov/transitionsParametersDry_glm.csv', sep=' ', header=None, names=['variable', 'value'])
    transitionsParametersDry.index += 1

    transitionsParametersWet = pd.read_csv('../results/visibleMarkov/transitionsParametersWet_glm.csv', sep=' ',
                                           header=None, names=['variable', 'value'])
    transitionsParametersWet.index += 1

    amountParametersGamma = pd.read_csv('../results/visibleMarkov/amountGamma_vglm.csv', sep=' ', header=None,
                                        names=['variable', 'log_mu', 'log_shape'])
    amountParametersGamma.index += 1

    # Rainfall amount parameters( Gamma parameters)
    # Fit gamma before simulate.
    rainfall_simulator_glm_plus_gamma.fit_gamma()
    fittedGamma = pd.read_csv('../results/visibleMarkov/fittedGamma.csv', index_col=0)

    #print('\n * Intercept means firts month (January) ')

    return transitionsParametersDry, transitionsParametersWet, amountParametersGamma, fittedGamma


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

        # Transitions probabilites.
        transitionsParametersDry = pd.read_csv('../results/visibleMarkov/transitionsParametersDry_glm_noenso.csv', sep=' ',
                                               header=None, names=['variable', 'value'])
        transitionsParametersDry.index += 1

        transitionsParametersWet = pd.read_csv('../results/visibleMarkov/transitionsParametersWet_glm_noenso.csv', sep=' ',
                                               header=None, names=['variable', 'value'])
        transitionsParametersWet.index += 1

        amountParametersGamma = pd.read_csv('../results/visibleMarkov/amountGamma_vglm_noenso.csv', sep=' ', header=None,
                                            names=['variable', 'log_mu', 'log_shape'])
        amountParametersGamma.index += 1

        # Run simulator.
        simulationStartDate, rainfallPerIteration, wetDaysPerIteration = rainfall_simulator_glm_plus_vglm_noenso.totalRun(simulationDataFrame=simulationDataFrame,
                                                                                                              transitionsParametersDry=transitionsParametersDry,
                                                                                                              transitionsParametersWet=transitionsParametersWet,
                                                                                                                amountParametersGamma=amountParametersGamma,
                                                                                                              iterations=1000)

    elif model == 'glm_vglm_deltaIndex':
        ### Load transitions and amount parameters.

        # Transitions probabilites.
        transitionsParametersDry = pd.read_csv('../results/visibleMarkov/transitionsParametersDry_glm_deltaIndex.csv', sep=' ',
                                               header=None, names=['variable', 'value'])
        transitionsParametersDry.index += 1

        transitionsParametersWet = pd.read_csv('../results/visibleMarkov/transitionsParametersWet_glm_deltaIndex.csv', sep=' ',
                                               header=None, names=['variable', 'value'])
        transitionsParametersWet.index += 1

        amountParametersGamma = pd.read_csv('../results/visibleMarkov/amountGamma_vglm_deltaIndex.csv', sep=' ', header=None,
                                            names=['variable', 'log_mu', 'log_shape'])
        amountParametersGamma.index += 1

        # Run simulator.
        simulationStartDate, rainfallPerIteration, wetDaysPerIteration = rainfall_simulator_glm_plus_vglm_deltaIndex.totalRun(simulationDataFrame=simulationDataFrame,
                                                                                                                              transitionsParametersDry=transitionsParametersDry,
                                                                                                                              transitionsParametersWet=transitionsParametersWet,
                                                                                                                              amountParametersGamma=amountParametersGamma,
                                                                                                                              iterations=1000)

    elif model == 'glm_vglm':
        ### Load transitions and amount parameters.

        # Transitions probabilites.
        transitionsParametersDry = pd.read_csv('../results/visibleMarkov/transitionsParametersDry_glm.csv', sep=' ',
                                               header=None, names=['variable', 'value'])
        transitionsParametersDry.index += 1

        transitionsParametersWet = pd.read_csv('../results/visibleMarkov/transitionsParametersWet_glm.csv', sep=' ',
                                               header=None, names=['variable', 'value'])
        transitionsParametersWet.index += 1

        amountParametersGamma = pd.read_csv('../results/visibleMarkov/amountGamma_vglm.csv', sep=' ', header=None,
                                            names=['variable', 'log_mu', 'log_shape'])
        amountParametersGamma.index += 1

        # Run simulator.
        simulationStartDate, rainfallPerIteration, wetDaysPerIteration = rainfall_simulator_glm_plus_vglm.totalRun(simulationDataFrame=simulationDataFrame,
                                                                                                              transitionsParametersDry=transitionsParametersDry,
                                                                                                              transitionsParametersWet=transitionsParametersWet,
                                                                                                                amountParametersGamma=amountParametersGamma,
                                                                                                              iterations=1000)
    else:
        print('Please choose a valid model to simulate:\n\t-"basic"\n\t-"glm_gamma"\n\t-"glm_vgam"')

    print('Simulation for: ', simulationStartDate, ' is done.')

    return simulationStartDate, rainfallPerIteration, wetDaysPerIteration


if __name__ == "__main__":

    # Build dataset which contains historical precipitation for El dorado Airport.
    startYear = 1972
    endYear = 2015
    concatYearsPrecipitation(startYear, endYear)

    # Download ENSO forecast probabilities.
    startDate = '2004-04-01'
    endDate = '2018-06-01'
    dates = pd.date_range(startDate, end=endDate, freq='MS')
    totalENSOforecastDownloader(startDate, endDate)
    ensoForecast = pickle.load(open('../datasets/ensoForecastProb/ensoForecastProbabilities.pickle','rb'))

    # Create a dataset which correlates precipitation, indices, dates and ENSO probabilities.
    startDate = '2005-01-01'
    endDate = '2015-12-31'
    #total_dataset = total_dataset_constructor(startDate=startDate, endDate=endDate)


    # Read transitions parameters calculated in R. (Run 1.0_vgamfit before load files.)
    transitionsParametersDry, transitionsParametersWet, amountParametersGamma, fittedGamma = recover_Transitions_parameters_from_R()

    # Loop to iterate over dates and simulates raining.
    model = 'glm_vglm_deltaIndex'  # 'Please choose a valid model to simulate: -"basic" -"glm_gamma" -"glm_vgam" -"glm_gamma_noenso" -"glm_vgam_noenso" -"glm_vgam_deltaIndex"'

    # Load all possible dates to simulate.
    dates = pd.date_range(start=startDate, end = endDate, freq='MS')
    dates = list(dates.strftime('%Y-%m-%d'))

    # Initialize time
    startTime = time.time()

    simulations = Parallel(n_jobs=-1, verbose=10)(delayed(run_simulation_per_date)(date,model) for date in dates)

    # Calculate time
    currentTime = time.time() - startTime
    with open('../results/simulations/simulation_'+model+'.pickle', 'wb') as f:
        pickle.dump(simulations, f)

    print('Simulations for all dates have finished successfully.')
    # Logging time.
    print('The elapsed time over simulation is: ', currentTime, ' seconds.')

    # Comment to know how to open file.
    '''
    with open('../results/simulations/simulation_'+model+'.pickle', 'rb') as f:
        data = pickle.load(f)
    '''

    ### Export to excel.

    # Open simulations.
    with open('../results/simulations/simulation_'+model+'.pickle', 'rb') as f:
        simulation = pickle.load(f)

    iterations = ['iteration_'+str(i) for i in range(1,1001)]

    precipitation = pd.DataFrame(columns=iterations)
    wetDays = pd.DataFrame(columns=iterations)
    for date in simulation:
        precipitation.loc[date[0]] = date[1]
        wetDays.loc[date[0]] = date[2]

    # Export dataframes to excel.
    precipitation.to_excel('../results/simulations/precipitation_simulation_'+model+'.xlsx')
    wetDays.to_excel('../results/simulations/wetdays_simulation_' + model + '.xlsx')


