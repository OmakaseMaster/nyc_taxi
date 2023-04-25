import pandas as pd
import matplotlib.pyplot as plt

class Zone_seperation:
    """ A class used to help seperate observations into different zones.

    input the dataframe and expected number of zones,
    output the original dataset with the expected zone seperation,
    and the distribution of numbers in zones in histogram.
    Also helps in visualization of zones.
    """
    def __init__(self, df, number_of_zones):
        self.df = df
        self.noz = number_of_zones

    
    def zone_seperation_helper(self, left, right, dis_lat, dis_lon):
        while right - left > 1e-3:
            mid = (left + right) / 2
            cur_zone = (dis_lat // mid + 1) * (dis_lon // mid + 1)

            if cur_zone == self.noz:
                return (mid, int(dis_lat // mid) + 1, int(dis_lon // mid) + 1)
            
            if cur_zone < self.noz:
                right = mid
            else:
                left = mid
        return (left, int(dis_lat // left) + 1, int(dis_lon // left) + 1)


    def zone_seperation(self):
        max_longitude = max(max(self.df["pickup_longitude"]), 
                            max(self.df["dropoff_longitude"]))
        max_latitude = max(max(self.df["pickup_latitude"]), 
                            max(self.df["dropoff_latitude"]))
        min_longitude = min(min(self.df["pickup_longitude"]), 
                            min(self.df["dropoff_longitude"]))
        min_latitude = min(min(self.df["pickup_latitude"]), 
                            min(self.df["dropoff_latitude"]))

        distance_longitude = max_longitude - min_longitude
        distance_latitude = max_latitude - min_latitude

        # binary search
        min_dis, max_dis = 1e-5, max(distance_longitude, distance_latitude)
        expected_dis, number_of_lat, number_of_lon = self.zone_seperation_helper(min_dis,
                                                                                 max_dis,
                                                                                 distance_latitude,
                                                                                 distance_longitude)
        return (expected_dis, number_of_lat, number_of_lon)
    

    def zone_marker(self):
        max_lon = max(self.df["pickup_longitude"])
        max_lat = max(self.df["pickup_latitude"])
        min_lon = min(self.df["pickup_longitude"])
        min_lat = min(self.df["pickup_latitude"])

        dis, no_lat, no_lon = self.zone_seperation()
        self.no_area = no_lat * no_lon

        lat_barrier = [min_lat + i * dis for i in range(no_lat)] + [max_lat + 0.001]
        lon_barrier = [min_lon + i * dis for i in range(no_lon)] + [max_lon + 0.001]

        def count_helper(x, barrier):
            ans = 0
            for b in barrier:
                if x >= b:
                    ans += 1
            return ans
        
        self.df["pickup_latitude_count"] = self.df["pickup_latitude"].apply(lambda x: count_helper(x, lat_barrier))
        self.df["pickup_longitude_count"] = self.df["pickup_longitude"].apply(lambda x: count_helper(x, lon_barrier))
        self.df["dropoff_latitude_count"] = self.df["dropoff_latitude"].apply(lambda x: count_helper(x, lat_barrier))
        self.df["dropoff_longitude_count"] = self.df["dropoff_longitude"].apply(lambda x: count_helper(x, lon_barrier))

        self.df["pickup_zone"] = self.df["pickup_latitude_count"] * no_lon + self.df["pickup_longitude_count"]
        self.df["dropoff_zone"] = self.df["dropoff_latitude_count"] * no_lon + self.df["dropoff_longitude_count"]
        return 
    

    def zone_counter(self):
        self.pickup_count = [0] * self.no_area
        self.dropoff_count = [0] * self.no_area

        pickup_dict = self.df["pickup_zone"].value_counts().to_dict()
        dropoff_dict = self.df["dropoff_zone"].value_counts().to_dict()

        for i in range(self.no_area):
            if i in pickup_dict.keys():
                self.pickup_count[i] = pickup_dict[i]
            if i in dropoff_dict.keys():
                self.dropoff_count[i] = dropoff_dict[i]
        x = [i for i in range(self.no_area)]

        plt.plot(x, self.pickup_count, label='Pickup zones')
        plt.plot(x, self.dropoff_count, label='Dropoff zones')
        plt.xlabel('Zone index')
        plt.ylabel('Number of rides')
        plt.legend()
        plt.show()


    def finilize(self):
        self.zone_marker()
        self.zone_counter()
        return self.df