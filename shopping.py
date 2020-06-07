import csv
import sys
import pandas as pd
import numpy as numpy

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


#after importing the data using pandas I got familiar with the data doing the following:
# 1. checked columns in resulting dataframe: data.columns
# 2. checked length number of rows and columns: data.shape
# 3. pre-checked the unique values for each columns; especially for columns Month, VisitorType, Weekend and Revenue as these had to be replaced
#       NOTE: abbreviation for June was June and not Jun! --> mapping-dictionary was layed out accordingly
#       for column in data.columns:
#           print(f'{len(data[column].unique())} values in column name {column}: \n{data[column].unique()}')
#           print(f'\n')
# 4. checked if there were any nan-values: data[data.isna().any(axis=1)]
#
#ALTERNATIVE to pandas would be: 
#with open('shopping.csv', newline='') as f:
#    reader = csv.reader(f)
#    data = list(reader)
    
def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # import data using pandas
    data = pd.read_csv('shopping.csv', header=0)

    #create dictionary that maps month names to month number (0-indexed)
    d = {'Jan':0, 'Feb':1, 'Mar':2, 'Apr':3, 'May':4,'June':5, 'Jul':6, 'Aug':7,
        'Sep':8, 'Oct':9, 'Nov':10, 'Dec':11}

    #change month abbreviations to integer values
    data.Month = data.Month.map(d)

    #change visitor values to 1 for returning visitor and 0 for non-returning visitor
    #NOTE: mapping dictionary not used because apart from 'returning' and 'new' visitors values there were also 'Others' --> using lambda if else statement can be used to change values
    data.VisitorType = data.VisitorType.map(lambda x : 1 if x=='Returning_Visitor' else 0)

    #change boolean values in Weekend to 1 or 0
    #alternative to .map() would be .replace(): data.Weekend = data.Weekend.replace([True, False],[1, 0])
    data.Weekend = data.Weekend.map(lambda x : 1 if x==True else 0)

    #change boolean values in Revenue to 1 or 0
    data.Revenue = data.Revenue.map(lambda x : 1 if x==True else 0)

    #define integer and float columns for check-up of dtype below
    ints = ['Administrative', 'Informational', 'ProductRelated', 'Month', 'OperatingSystems', 
            'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend'] 
    floats = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 
            'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']

    #check type of integer-columns and convert if not of integer-type
    for value in ints:
        if data[value].dtype != 'int64':
            # print(f'values in column {value.upper()} changed to int (was {data[value].dtype} before \n') #for troubleshooting
            data = data.astype({value: 'int64'})
        else:
            continue

    #check type of float-columns and convert if not of float-type  
    for value in floats:
        if data[value].dtype != 'float64':
            # print(f'values in column {value.upper()} changed to float (was {data[value].dtype} before \n') #for troubleshooting
            data = data.astype({value: 'float64'})
        else:
            continue

    #evidence-values: create list of lists from dataframe
    evidence = data.iloc[:,:-1].values.tolist()
    #label-values: create list from last column (series)
    labels = data.iloc[:,-1].values.tolist()

    #check if evidence and labels are of same length
    if len(evidence) != len (labels):
        print('ERROR! Evidence and label lists not of same length. Check code!')
    else:
        print(f'There are {len(evidence)} entries in this dataset.\n')
    
    #return evidence and labels as tuple
    return (evidence, labels)  


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # create classifier implementing the k-nearest neighbors
    model = KNeighborsClassifier(n_neighbors=1)

    #train classifier named model
    model.fit(evidence, labels)

    #return trained classifier
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # get number of actual positives from labels
    positives = labels.count(1)

    # get number of actual negatives from labels
    negatives = labels.count(0)

    #initiate sensitivity and specificity variables
    sens = 0
    spec = 0

    #iterate over actual labels and predicted labels at the same time (use zip as the two lists should be of same length)
    for label, pred in zip(labels, predictions):
        if label == 1:
            # if prediction is correct, increase sensitivity counter by one
            if label == pred:
                sens += 1
        else:
            # if prediction is correct, increase specificity counter by one
            if label == pred:
                spec += 1
    
    #calculate proportion of correct positive and negatives
    sensitivity = sens / positives
    specificity = spec / negatives

    #return both values
    return (sensitivity, specificity)

if __name__ == "__main__":
    main()
