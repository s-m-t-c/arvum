import pandas as pd
import numpy as np
data = pd.read_csv("/g/data/r78/LCCS_Aberystwyth/training_data/cultivated/2010_2015_training_data_combined_17022021/2010_2015_median_training_data.txt", header=0, sep=" ")
data['binary_class'] = np.where(data['classnum'] == 111, 1, 0)
data.to_csv("2010_2015_median_training_data_binary.txt", sep=" ", index=False)
