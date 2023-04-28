import pandas as pd
import matplotlib.pyplot as plt

class Zone_selection:
    """ A class used to do zone selection so that zones with too few data points
    can be omitted.
    """
    def __init__(self, df, cutoff, number_of_zone):
        self.df = df
        self.cutoff = cutoff
        self.noz = number_of_zone
        self.zone_count()
        self.zone_present()

    
    def zone_count(self):
        self.pickup_count = [0] * self.noz
        self.dropoff_count = [0] * self.noz

        pickup_dict = self.df["pickup_zone"].value_counts().to_dict()
        dropoff_dict = self.df["dropoff_zone"].value_counts().to_dict()
        for i in range(self.noz):
            if i in pickup_dict.keys():
                self.pickup_count[i] = pickup_dict[i]
            if i in dropoff_dict.keys():
                self.dropoff_count[i] = dropoff_dict[i]

        self.zone_count = pd.DataFrame({"pickup": self.pickup_count,
                                        "dropoff": self.dropoff_count})
        return
        
    
    def zone_select(self):
        qualified_zone_set = set(self.zone_count[(self.zone_count['pickup'] > self.cutoff) \
                                                 & (self.zone_count['dropoff'] > self.cutoff)].index)
        print("Number of selected zones: ", len(qualified_zone_set))

        self.result_df = self.df[(self.df['pickup_zone'].isin(qualified_zone_set)) \
                                  & (self.df['dropoff_zone'].isin(qualified_zone_set))]
        return qualified_zone_set
    

    def zone_present(self):
        # recode zone number
        selected_index = list(self.zone_select())
        selected_index.sort()
        new_index = [i for i in range(len(selected_index))]
        recode_index = dict(zip(selected_index, new_index))

        self.result_df['pickup_zone'] = self.result_df['pickup_zone'].map(recode_index)
        self.result_df['dropoff_zone'] = self.result_df['dropoff_zone'].map(recode_index)

        # present histogram
        self.pickup_count = [0] * len(selected_index)
        self.dropoff_count = [0] * len(selected_index)

        pickup_dict = self.result_df["pickup_zone"].value_counts().to_dict()
        dropoff_dict = self.result_df["dropoff_zone"].value_counts().to_dict()

        for i in range(len(selected_index)):
            if i in pickup_dict.keys():
                self.pickup_count[i] = pickup_dict[i]
            if i in dropoff_dict.keys():
                self.dropoff_count[i] = dropoff_dict[i]
        x = [i for i in range(len(selected_index))]

        plt.plot(x, self.pickup_count, label='Pickup zones')
        plt.plot(x, self.dropoff_count, label='Dropoff zones')
        plt.xlabel('Zone index')
        plt.ylabel('Number of rides')
        plt.legend()
        plt.show()
        return 