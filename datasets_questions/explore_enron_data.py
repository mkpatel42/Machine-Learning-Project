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

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))


# How many data points (people) are in the dataset?
print(len(enron_data))
# For each person, how many features are available?
print(len(enron_data['ELLIOTT STEVEN']))
# How many POIs are there in the E+F dataset?
count = 0
for key in enron_data.keys():
    if enron_data[key]['poi'] == 1:
        count+= 1 ;

print(count)


