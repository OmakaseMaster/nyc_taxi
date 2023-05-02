import pandas as pd
import numpy as np
import bisect
import random
import math
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import time
from numba import jit
from itertools import product
import multiprocessing as mp
from multiprocessing import Pool, Array
from ctypes import c_double, c_int, c_long
from functools import partial
import sys
import pickle
# global variable:
data = pd.read_csv("nyc_data_cleaned.csv")
def main():
    init_time = time.time()
    random.seed(0)
    ## read inputs:
    number_of_zone = len(data["pickup_zone"].unique())
    num_cores = mp.cpu_count()
    ## Time simulation:
    start_time = time.time()
    time_arr = Array(c_double, number_of_zone * number_of_zone * 60*24)
    ijs = list(product(range(number_of_zone), range(number_of_zone))) # all possible pairs of i,j
    random_uniform = np.random.uniform(0, 1, size=(number_of_zone, number_of_zone, 60))
    # > pre-compute all uniform distribution at once
    time_simulation_partial = partial(time_simulation, random_uniform) # fix 1st,2nd arg
    with Pool(processes=num_cores, initializer=init, initargs=(time_arr,)) as pool:
        pool.map(time_simulation_partial, ijs)
    simulated_time = np.frombuffer(time_arr.get_obj())
    simulated_time = simulated_time.reshape((number_of_zone, number_of_zone, 60*24))
    end_time = time.time()
    print(f"Time took for time simulation: {end_time-start_time}")
    ## Trip simulation
    # multiprocessing didn't improve efficiency...
    trip_arr = Array(c_long, number_of_zone * number_of_zone * 60*24)
    # trip_simulation_partial = partial(trip_simulation, data)
    with Pool(processes=num_cores, initializer=init2, initargs=(trip_arr,)) as pool:
        pool.map(trip_simulation, ijs)
    simulated_trip = np.frombuffer(trip_arr.get_obj())
    simulated_trip = simulated_trip.reshape((number_of_zone, number_of_zone, 60*24))
    print(f"Time took for trip simulation: {time.time()-end_time}")
    ## Ride fare simulation
    beta0, beta1, beta2 = linear_model()
    start_time = time.time()
    simulated_distance = dist_simulation()
    end_time = time.time()
    print(f"Time took for distance simulation: {end_time-start_time}")
    wait_time = wait_simulation(simulated_trip)
    simulated_fare = fare_simulation(simulated_distance, simulated_time, beta0, beta1, beta2)
    print(f"Time took for distance & wait time simulation: {time.time()-end_time}")
    setups = {"simulated_time":simulated_time, 
              "simulated_distance":simulated_distance,
              "simulated_trip":simulated_trip,
              "wait_time":wait_time,
              "simulated_fare":simulated_fare
                }
    savefile = open(f'simulated_values.pkl', 'wb')
    pickle.dump(setups, savefile)
    savefile.close()
    print(f"Finish simulation setup, total time spent: {time.time()-init_time}")
def fare_simulation(distance, time, beta0, beta1, beta2):
    # get all fares at once by numpy broadcasting
    return beta0 + beta1 * distance[:,:,None] + beta2 * time
def time_simulation(random_uniform, ij, number_of_zone=40):
    with time_arr.get_lock():  # every process is accessing the same slice, so get_lock()
        simulated_time = np.frombuffer(time_arr.get_obj())
        simulated_time = simulated_time.reshape((number_of_zone, number_of_zone, 60*24))
        # > reshape for array operation
        i, j = ij[0], ij[1]
        df = data[(data["pickup_zone"] == i)
                  & (data["dropoff_zone"] == j)]
        average_time = df["trip_time_in_secs"].mean()
        # t stands for hour information
        for t in range(24):
            # df_sub includes only the rides from zone i to zone j during t:t+1 hour
            df_sub = df[(df["pickup_hour"] == t)]
            # get all the travel time for rides
            travel_time = np.array(df_sub["trip_time_in_secs"]).astype(np.int64)
            # if no data is provided, use the average travel time of the day
            if len(travel_time) == 0:
                simulated_time[i][j][t * 60: (t + 1) * 60] = average_time
            else:
                # simulate ecdf on given time for rides
                ecdf = np.cumsum(np.bincount(travel_time)) / len(travel_time)
                interp_func = interp1d(ecdf,
                                       np.arange(len(ecdf)),
                                       kind='linear',
                                       bounds_error=False,
                                       fill_value='extrapolate')
                # generate 60 random samples from the fitted distribution
                # so that each trips start from each minute is simulated in this hour
                samples = interp_func(random_uniform[i,j,:])
                simulated_time[i][j][t * 60: (t + 1) * 60] = samples
def trip_simulation(ij, number_of_zone=40):
    with trip_arr.get_lock():  # every process is accessing the same slice, so get_lock()
        simulated_trip = np.frombuffer(trip_arr.get_obj())  
        simulated_trip = simulated_trip.reshape((number_of_zone, number_of_zone, 60*24))
        i, j = ij[0], ij[1]
        for t in range(24):
            # df_sub includes only the rides from zone i to zone j during t:t+1 hour
            df_sub = data[(data["pickup_zone"] == i)
                          & (data["dropoff_zone"] == j)
                          & (data["pickup_hour"] == t)]
            # get all the travel time for rides
            travel_count = len(df_sub)
            samples = np.random.multinomial(travel_count, np.ones(60) / 60)
            simulated_trip[i][j][t * 60: (t + 1) * 60] = samples
def linear_model():
    global data
    fare = np.array(data["total_amount"])
    distance = np.array(data["trip_distance"]).reshape((-1, 1))
    travel_time = np.array(data["trip_time_in_secs"]).reshape((-1, 1))
    x = np.concatenate((distance, travel_time), axis=1)
    # create a linear regression model and fit it to the data
    fare_model = LinearRegression().fit(x, fare)
    # calculate R-squared
    r_squared = fare_model.score(x, fare)
    print("The R squared of the model is: ", r_squared)
    beta0 = fare_model.intercept_
    beta1 = fare_model.coef_[0]
    beta2 = fare_model.coef_[1]
    return beta0, beta1, beta2
def dist_simulation(number_of_zone=40):
    global data
    grouped_df = data.groupby(["pickup_zone", "dropoff_zone"]) # group data by zone
    simulated_distance = grouped_df["trip_distance"].apply(lambda x: random.gauss(np.mean(x), np.std(x)))
    # > calcualte the simulated_distance within each group
    simulated_distance = np.array(simulated_distance).reshape((number_of_zone,number_of_zone))
    return simulated_distance # turn to np.array
def wait_simulation(simulated_trip, number_of_zone=40):
    wait_time = np.zeros((number_of_zone, 60 * 24), dtype=np.int32)
    for i in range(number_of_zone):
        have_trip = np.any(simulated_trip[i], axis=0)
        prev = np.where(have_trip)[0][-1] if np.any(have_trip) else -1
        for k in range(60 * 24 - 1, -1, -1):
            if have_trip[k]:
                prev = k
            elif prev == -1:
                wait_time[i][k] = -1
            else:
                wait_time[i][k] = prev - k
    return wait_time
def init(shared_arr_):    # to define the global variable in each multi-process
    global time_arr
    time_arr = shared_arr_
def init2(shared_arr_2):   # same as above but for another pool
    global trip_arr
    trip_arr = shared_arr_2
if __name__ == "__main__":
    main()

    
    
