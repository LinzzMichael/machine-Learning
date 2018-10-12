#!/usr/bin/python
import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    max = 0
    error = abs(predictions - net_worths)
    # print(error)
    for i in range(0,9):
        max = 0
        index = 0
        for j in range(0,len(ages)):
            if error[j] > max:
                index = j
                max = error[j]
        error = np.delete(error, index, 0)
        ages = np.delete(ages, index, 0)
        net_worths = np.delete(net_worths, index, 0)
    # print(ages.shape)
    for i in range(0,len(ages)):
        temp = (ages[i], net_worths[i], error[i])
        cleaned_data.append(temp)





    
    return cleaned_data

