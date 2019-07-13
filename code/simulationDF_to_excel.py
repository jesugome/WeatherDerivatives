import pickle
import pandas as pd

if __name__ == "__main__":

    # Define model to export.
    model = 'glm_gamma'  # 'Please choose a valid model to simulate: -"basic" -"glm_gamma" -"glm_vgam"'

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