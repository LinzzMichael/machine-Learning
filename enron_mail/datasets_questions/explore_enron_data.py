#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
num_Salary = 0
num_Email = 0

for key in enron_data:
    if enron_data[key]["salary"] != "NaN":
        num_Salary = num_Salary + 1
    if enron_data[key]["email_address"] != "NaN":
        num_Email = num_Email + 1
print(num_Salary)
print(num_Email)
    
print(enron_data["SKILLING JEFFREY K"])


