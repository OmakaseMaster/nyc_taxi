# Project

This study is deginated to reveal the average income for taxi drivers in Manhattan by simulating taxi rides, and to compare between different strategies

# Data

## taxi_data.csv

Raw data

## nyc_data_cleans.csv

Cleaned data from taxi_preprocessing.ipynb

## simulated_values.pkl

A dictionary that contains simulation setup info. Including time, 
distance, trip, wait time, fare. Created from simulation_setup.py

# Notebooks

## taxi_preprocessing.ipynb

The ipynb file is the well documented process for data wrangling. It contains data import and zone-seperation and zone-selection part and helps rewrite a cleaned data set for next-step analysis.

## taxi_simulation.ipynb

The ipynb file is the well documented process for simulation towards trip count and travel time.

## accelarated_taxi_simulation.ipynb

The ipynb file ran the simulation of different strategies.

# Helper programs

## ZoneSeperation.py

A helper class to do zone seperation.

## ZoneSelection.py

A helper class to do zone selection which omits zones with too few rides and not in Manhattan area.

## simulation_setup.py

Set up the needed environment for simulation process

## simulation_func.py

Simulation functions of different strategies

**Strategy 1**: Driver 1 prefers to start at an initial zone, waiting for the first-come trip. After completing the current trip, Driver 1 will stay at the new location and repeat the same process.  
**Strategy 2**: Driver 2 starts at an initial zone, waiting for the first-come trip. He plans to choose the trip with maximum distance. After completing the current trip, Driver 2 will stay at the new location and repeat the same process.  
**Strategy 3**: Driver 3 starts at an initial zone, waiting for the first-come trip. He plans to choose the trip with minimum distance. After completing the current trip, Driver 3 will stay at the new location and repeat the same process.  
