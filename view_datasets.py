import pandas as pd
import numpy as np
import sys
sys.path.insert(1, '/home/jahan/Missingness')
import UAT.datasets as data

data_list = data.get_list(0, key=12, test=lambda x, m: x == m)
data_list = data_list.reset_index().sort_values(by=['NumberOfInstances'])
class_filter = np.array([x == "Supervised Classification" for x in data_list.task_type])
selection = np.arange(len(data_list))[class_filter]
print(data_list[
            ['name', 'NumberOfInstances', 'NumberOfFeatures', 'NumberOfClasses', 'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures']
            ].loc[data_list.index[selection]])
