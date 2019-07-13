# Import needed libraries.
import pandas as pd
import datetime
import pickle
import os


def oneYearMonthDonwloader(date):
    url = ''

    if date <= datetime.datetime(2014, 1, 1):
        month = date.strftime('%m')
        year = date.year
        # Build complete URL.
        url = 'https://iri.columbia.edu/our-expertise/climate/forecasts/enso/archive/' + str(year) + str(
            month) + '/figure3.html'

        # Download Data
        yearMonthENSOForecast = pd.read_html(url, header=0)
        return yearMonthENSOForecast[1]
    else:
        month = date.strftime('%B')
        year = date.year

        # Build complete URL.
        url = 'https://iri.columbia.edu/our-expertise/climate/forecasts/enso/' + str(year) + '-' + str(
            month) + '-quick-look/?enso_tab=enso-iri_plume'
        # Download Data
        yearMonthENSOForecast = pd.read_html(url, header=0)
        return yearMonthENSOForecast[2]


def totalENSOforecastDownloader(startDate, endDate):

    if not os.path.exists("../datasets/ensoForecastProb/ensoForecastProbabilities.pickle"):
        dates = pd.date_range(startDate, end=endDate, freq='MS')
        ensoForecast = {}

        print('Downloading data from: ...')

        for date in dates:
            month = date.strftime('%m')
            year = date.year

            storeStr = str(year) + '-' + str(month)
            print(storeStr)

            # Store in dictionary --> '2002-04'
            ensoForecast[storeStr] = oneYearMonthDonwloader(date)

        # Save data in pickle format.
        if not os.path.exists('../datasets/ensoForecastProb/'):
            os.mkdir('../datasets/ensoForecastProb/')
        pickle.dump(ensoForecast, open("../datasets/ensoForecastProb/ensoForecastProbabilities.pickle", "wb"))

        return ensoForecast


if __name__ == "__main__":

    # Download enso forecast from IRI webpage.

    # Generate a timestamp with all months to download data.
    startDate = '2004-04-01'
    endDate = '2018-06-01'
    dates = pd.date_range(startDate, end=endDate, freq='MS')
    ensoForecast = totalENSOforecastDownloader(startDate, endDate)
