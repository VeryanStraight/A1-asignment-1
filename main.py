import pandas as pd
import numpy as np
import sys
import statistics as stat

k = 3


def find_range(column):
    return column.max() - column.min()


def make_prediction(row):
    # drop the actual class in the test data
    data_row = row.drop(labels=['Class'])

    # crate an array containing the difference for each row in the training data
    difference_from_training = np.linalg.norm((X - data_row.values)/range_data, axis=1)

    # sort by distance and get the nearst K neighbours
    nearest_neighbors = difference_from_training.argsort()[:k]

    # get the classes of these nearest neighbours
    nearest_neighbor_classes = y[nearest_neighbors]

    # set the class to the mode of the nearst neighbours
    predicted_class = stat.mode(nearest_neighbor_classes)
    row['Class'] = predicted_class
    return row


# use pandas to read in data
test_data = pd.read_csv(sys.argv[1], sep=' ')
training_data = pd.read_csv(sys.argv[2], sep=' ')

test_data.to_excel('test_data.xlsx')
training_data.to_excel('training_data.xlsx')

training_data_without_class = training_data.drop("Class", axis=1)
training_data_classes = training_data["Class"]

# convert to 2d arrays
X = training_data_without_class.values
y = training_data_classes.values

range_data = training_data_without_class.apply(find_range, axis=0).values
print(range_data)

# apply knn
prediction_data = test_data.apply(make_prediction, axis=1)

# output
prediction_correct = prediction_data['Class'] == test_data['Class']
proportion = prediction_correct.value_counts(normalize=True)
print('----------------')

prediction_data['Original class'] = test_data['Class']
prediction_data.to_excel('output knn.xlsx')
print(prediction_data['Class'].tolist())
print(proportion)
