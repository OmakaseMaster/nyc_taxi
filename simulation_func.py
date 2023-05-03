import pandas as pd
import numpy as np
import random
import math
import time
import sys
import pickle
def load_py(location):   # load the simulation setup values
    file = open(f'{location}', 'rb')     
    data = pickle.load(file)
    file.close()
    return data
def available_trip(i, k, simulated_trip):
    ans = np.where(simulated_trip[i, :, k])[0]  # use [0] to extract the idx from np.where
    return ans 
import datetime 
import datetime 

def driver_sim1(simulated_values,i_start,k_start = 8*60,k_end = 22*60, verbose=0, seed=0):
    """
    simulated_values: simulation setup
    i_start: initial working zone
    k_start, k_end: start/end time in minutes
    verbose: no. of iteration per report 
    return: dict, zone frequency, total # of trips & amount of fares
    """
    random.seed(seed)  # set specific seed for reproduction
    simulated_time = simulated_values["simulated_time"]
    simulated_distance = simulated_values["simulated_distance"]
    simulated_trip = simulated_values["simulated_trip"]
    simulated_fare = simulated_values["simulated_fare"]
    wait_time = simulated_values["wait_time"]
    if verbose != 0:
        print(f'{str(datetime.timedelta(minutes=int(k_start)))} : Driver start working at zone {i_start}')
    # initialization
    cur_i = i_start
    cur_k = k_start
    cur_fare = 0
    cur_case = 0
    rest1 = 0
    rest2 = 0
    cnt = 0 # cnt iteration
    zone_freq = np.zeros(40) # buffer for saving 40 zones frequency
    while cur_k <= k_end:
        cnt += 1
        # trip quiry
        # set initial waiting time as 0
        wt = 0
        while len(available_trip(cur_i, cur_k, simulated_trip)) == 0:
            wt = wait_time[cur_i][cur_k]
            cur_k += wt
        j = random.choice(available_trip(cur_i, cur_k, simulated_trip))
        zone_freq[j] += 1   # +1 for the destination zone
        sim_dis = simulated_distance[cur_i, j]
        # trip ongoing
        cur_k += math.ceil(simulated_time[cur_i][j][cur_k]/60)
        cur_i = j
        cur_fare +=  simulated_fare[cur_i][j][cur_k]
        if verbose == 0:
            pass
        elif (cnt % verbose == 0):
            print(f'{str(datetime.timedelta(minutes=int(cur_k - wt)))} : New trip from zone {cur_i} to zone {j} assigned!!!')
            print(f'{str(datetime.timedelta(minutes=int(cur_k)))} : New trip from zone {cur_i} to zone {j} started!!!')
            print(f'{str(datetime.timedelta(minutes=int(cur_k)))} : Trip completed at {cur_i},Total fare = $ {cur_fare}')
        # rest time
        if cur_k > 12*60 and rest1 == 0:
            cur_k += 30 
            if verbose != 0:
                print(f'{str(datetime.timedelta(minutes=int(cur_k)))} : Rest for 30 min ')
            rest1 = 1
        if cur_k > 16*60 + 30 and rest2 == 0:
            cur_k += 30 
            if verbose != 0:
                print(f'{str(datetime.timedelta(minutes=int(cur_k)))} : Rest for 30 min ')
            rest2 = 1
        cur_case += 1
    sep = '-'
    if verbose != 0:
        print(f'From {str(datetime.timedelta(minutes=int(k_start)))} to {str(datetime.timedelta(minutes=int(cur_k)))} : {cur_case} trips finished ($ {cur_fare}!!!)\n{sep*30}')
    output = {'zone_freq':zone_freq, "num_of_trip":cur_case, "total_fare":cur_fare}
    return output

def driver_sim2(simulated_values,i_start,k_start = 8*60,k_end = 22*60, verbose=0, seed=0):
    """
    simulated_values: simulation setup
    i_start: initial working zone
    k_start, k_end: start/end time in minutes
    verbose: no. of iteration per report 
    return: dict, zone frequency, total # of trips & amount of fares
    """
    random.seed(seed)  # set specific seed for reproduction
    simulated_time = simulated_values["simulated_time"]
    simulated_distance = simulated_values["simulated_distance"]
    simulated_trip = simulated_values["simulated_trip"]
    simulated_fare = simulated_values["simulated_fare"]
    wait_time = simulated_values["wait_time"]
    if verbose != 0:
        print(f'{str(datetime.timedelta(minutes=int(k_start)))} : Driver start working at zone {i_start}')
    # initialization
    cur_i = i_start
    cur_k = k_start
    cur_fare = 0
    cur_case = 0
    rest1 = 0
    rest2 = 0
    cnt = 0 # cnt iteration
    zone_freq = np.zeros(40) # buffer for saving 40 zones frequency
    while cur_k <= k_end:
        cnt += 1
        # trip quiry
        # set initial waiting time as 0
        wt = 0
        while len(available_trip(cur_i, cur_k, simulated_trip)) == 0:
            wt = wait_time[cur_i][cur_k]
            cur_k += wt
        
        temp_trip = available_trip(cur_i, cur_k,simulated_trip)
        temp_dist = [simulated_distance[cur_i][trip] for trip in temp_trip]
        
        sim_dis = max(temp_dist) # max distance in the zone list
        idx = temp_dist.index(sim_dis)
        j = temp_trip[idx]

        zone_freq[j] += 1   # +1 for the destination zone
        # sim_dis = simulated_distance[cur_i, j]
        # trip ongoing
        cur_k += math.ceil(simulated_time[cur_i][j][cur_k]/60)
        cur_i = j
        cur_fare +=  simulated_fare[cur_i][j][cur_k]
        if verbose == 0:
            pass
        elif (cnt % verbose == 0):
            print(f'{str(datetime.timedelta(minutes=int(cur_k - wt)))} : New trip from zone {cur_i} to zone {j} assigned!!!')
            print(f'{str(datetime.timedelta(minutes=int(cur_k)))} : New trip from zone {cur_i} to zone {j} started!!!')
            print(f'{str(datetime.timedelta(minutes=int(cur_k)))} : Trip completed at {cur_i},Total fare = $ {cur_fare}')
        # rest time
        if cur_k > 12*60 and rest1 == 0:
            cur_k += 30 
            if verbose != 0:
                print(f'{str(datetime.timedelta(minutes=int(cur_k)))} : Rest for 30 min ')
            rest1 = 1
        if cur_k > 16*60 + 30 and rest2 == 0:
            cur_k += 30 
            if verbose != 0:
                print(f'{str(datetime.timedelta(minutes=int(cur_k)))} : Rest for 30 min ')
            rest2 = 1
        cur_case += 1
    sep = '-'
    if verbose != 0:
        print(f'From {str(datetime.timedelta(minutes=int(k_start)))} to {str(datetime.timedelta(minutes=int(cur_k)))} : {cur_case} trips finished ($ {cur_fare}!!!)\n{sep*30}')
    output = {'zone_freq':zone_freq, "num_of_trip":cur_case, "total_fare":cur_fare}
    return output

def driver_sim3(simulated_values,i_start,k_start = 8*60,k_end = 22*60, verbose=0, seed=0):
    """
    simulated_values: simulation setup
    i_start: initial working zone
    k_start, k_end: start/end time in minutes
    verbose: no. of iteration per report 
    return: dict, zone frequency, total # of trips & amount of fares
    """
    random.seed(seed)  # set specific seed for reproduction
    simulated_time = simulated_values["simulated_time"]
    simulated_distance = simulated_values["simulated_distance"]
    simulated_trip = simulated_values["simulated_trip"]
    simulated_fare = simulated_values["simulated_fare"]
    wait_time = simulated_values["wait_time"]
    if verbose != 0:
        print(f'{str(datetime.timedelta(minutes=int(k_start)))} : Driver start working at zone {i_start}')
    # initialization
    cur_i = i_start
    cur_k = k_start
    cur_fare = 0
    cur_case = 0
    rest1 = 0
    rest2 = 0
    cnt = 0 # cnt iteration
    zone_freq = np.zeros(40) # buffer for saving 40 zones frequency
    while cur_k <= k_end:
        cnt += 1
        # trip quiry
        # set initial waiting time as 0
        wt = 0
        while len(available_trip(cur_i, cur_k, simulated_trip)) == 0:
            wt = wait_time[cur_i][cur_k]
            cur_k += wt
        
        temp_trip = available_trip(cur_i, cur_k,simulated_trip)
        temp_dist = [simulated_distance[cur_i][trip] for trip in temp_trip]
        
        sim_dis = min(temp_dist) # min distance in the zone list
        idx = temp_dist.index(sim_dis)
        j = temp_trip[idx]

        zone_freq[j] += 1   # +1 for the destination zone
        # sim_dis = simulated_distance[cur_i, j]
        # trip ongoing
        cur_k += math.ceil(simulated_time[cur_i][j][cur_k]/60)
        cur_i = j
        cur_fare +=  simulated_fare[cur_i][j][cur_k]
        if verbose == 0:
            pass
        elif (cnt % verbose == 0):
            print(f'{str(datetime.timedelta(minutes=int(cur_k - wt)))} : New trip from zone {cur_i} to zone {j} assigned!!!')
            print(f'{str(datetime.timedelta(minutes=int(cur_k)))} : New trip from zone {cur_i} to zone {j} started!!!')
            print(f'{str(datetime.timedelta(minutes=int(cur_k)))} : Trip completed at {cur_i},Total fare = $ {cur_fare}')
        # rest time
        if cur_k > 12*60 and rest1 == 0:
            cur_k += 30 
            if verbose != 0:
                print(f'{str(datetime.timedelta(minutes=int(cur_k)))} : Rest for 30 min ')
            rest1 = 1
        if cur_k > 16*60 + 30 and rest2 == 0:
            cur_k += 30 
            if verbose != 0:
                print(f'{str(datetime.timedelta(minutes=int(cur_k)))} : Rest for 30 min ')
            rest2 = 1
        cur_case += 1
    sep = '-'
    if verbose != 0:
        print(f'From {str(datetime.timedelta(minutes=int(k_start)))} to {str(datetime.timedelta(minutes=int(cur_k)))} : {cur_case} trips finished ($ {cur_fare}!!!)\n{sep*30}')
    output = {'zone_freq':zone_freq, "num_of_trip":cur_case, "total_fare":cur_fare}
    return output

def main():
    # inputs:
    simulated_values = load_py("simulated_values.pkl")
    if sys.argv[1] == "one":
        # strategy 1:
        driver_sim1(simulated_values, 22, verbose=1)
    elif sys.argv[1] == "two":
        # strategy 2:
        driver_sim2(simulated_values, 22, verbose=1)
    elif sys.argv[1] == "three":
        # strategy 3:
        driver_sim3(simulated_values, 22, verbose=1)
    elif sys.argv[1] == "all_zones_once":
        for i_start in range(40):
            print("------------------------------------------------------------")
            print(f"Zone {i_start}:")
            print("Strategy 1:")
            print([driver_sim1(simulated_values, i_start, verbose=0)[key] for key in ["num_of_trip","total_fare"]])
            print("Strategy 2:")
            print([driver_sim2(simulated_values, i_start, verbose=0)[key] for key in ["num_of_trip","total_fare"]])
            print("Strategy 3:")
            print([driver_sim3(simulated_values, i_start, verbose=0)[key] for key in ["num_of_trip","total_fare"]])


if __name__ == "__main__":
    main()

    
    
