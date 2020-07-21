import pandas as pd
import numpy as np
data = pd.read_csv("2010_2015_median_training_data.txt", header=0, sep=" ")
data['binary_class'] = np.where(data['classnum'] == 111, 111, 0)
data.to_csv("2010_2015_median_training_data_binary.txt", sep=" ", index=False)
