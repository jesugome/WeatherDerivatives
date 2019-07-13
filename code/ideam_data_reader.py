# Import needed libraries.
import pandas as pd
import numpy as np
from io import StringIO
import os


def isleapyear(year):

    # """Determine whether a year is a leap year."""
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        return True
    return False


def loadYear(year):

    # Read only one year.
    year = str(year)

    # Configure path to read txts in loadyear function.
    path = '../datasets/ideamBogota/'
    filedata = open(path + year + '.txt', 'r')

    # Create a dataframe from the year's txt.

    columnNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    precipitationYear = pd.read_csv(StringIO('\n'.join(' '.join(l.split()) for l in filedata)), sep=' ', header=None,
                                    names=columnNames, skiprows=lambda x: x in list(range(0, 3)), skipfooter=4)

    # Sort data to solve problem of 28 days of Feb.

    for i in range(28, 30):
        for j in reversed(range(1, 12)):
            precipitationYear.iloc[i, j] = precipitationYear.iloc[i, j - 1]

        # Fix leap years.

        if isleapyear(int(year)) and i == 28:
            count = 1
        else:
            precipitationYear.iloc[i, 1] = np.nan

            # Fix problem related to months with 31 days.

    precipitationYear.iloc[30, 11] = precipitationYear.iloc[30, 6]
    precipitationYear.iloc[30, 9] = precipitationYear.iloc[30, 5]
    precipitationYear.iloc[30, 7] = precipitationYear.iloc[30, 4]
    precipitationYear.iloc[30, 6] = precipitationYear.iloc[30, 3]
    precipitationYear.iloc[30, 4] = precipitationYear.iloc[30, 2]
    precipitationYear.iloc[30, 2] = precipitationYear.iloc[30, 1]

    for i in [1, 3, 5, 8, 10]:
        precipitationYear.iloc[30, i] = np.nan

    return precipitationYear


def convertOneYearToSeries(dataFrameYear, nYear):

    # Convert one year data frame to timeseries.
    dataFrameYearT = dataFrameYear.T
    dates = pd.date_range(str(nYear) + '-01-01', end=str(nYear) + '-12-31', freq='D')

    dataFrameYearAllTime = dataFrameYearT.stack()

    dataFrameYearAllTime.index = dates

    return dataFrameYearAllTime


def plotAYear(testYear,nYear):
    # Plot data from one year.
    timeYear = convertOneYearToSeries(testYear, nYear)
    meanTimeYear = timeYear.mean()
    ax = timeYear.plot(title='Precipitation(mm) for ' + str(nYear), figsize=(20, 10), grid=True)
    ax.axhline(y=meanTimeYear, xmin=-1, xmax=1, color='r', linestyle='--', lw=2)


def concatYearsPrecipitation(startYear, endYear):
    # Concatenate all time series between a years range.

    if not os.path.exists('../datasets/precipitationAllTime.csv'):

        precipitationAllTime = loadYear(startYear)
        precipitationAllTime = convertOneYearToSeries(precipitationAllTime, startYear)

        for i in range(startYear + 1, endYear + 1):
            tempPrecipitation = loadYear(i)
            tempPrecipitation = convertOneYearToSeries(tempPrecipitation, i)

            precipitationAllTime = pd.concat([precipitationAllTime, tempPrecipitation])

        # Save precipitation data to .csv
        precipitationAllTime.to_csv('../datasets/precipitationAllTime.csv')
        return precipitationAllTime



def plotOverYears(startYear, endYear):
    precipitationAllTime = concatYearsPrecipitation(startYear, endYear)
    meanAllTime = precipitationAllTime.mean()
    ax = precipitationAllTime.plot(title='Precipitation(mm) from ' + str(startYear) + ' to ' + str(endYear),
                                   figsize=(20, 10), grid=True, color='steelblue')
    ax.axhline(y=meanAllTime, xmin=-1, xmax=1, color='r', linestyle='--', lw=2)


if __name__ == "__main__":

    # Read and build dataset which contains El dorado Airport precipitation.

    # Concatenate precipitation over years.
    startYear = 1972
    endYear = 2015
    precipitationAllTime = concatYearsPrecipitation(startYear, endYear)
