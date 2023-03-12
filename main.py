import pandas as pd
import numpy as np
import sys
import statistics as stat

k = 3


def make_prediction(row):
    data_row = row.drop(labels=['Class'])
    difference_from_training = np.linalg.norm(X - data_row.values, axis=1)
    nearest_neighbors = difference_from_training.argsort()[:k]
    nearest_neighbor_classes = y[nearest_neighbors]
    predicted_class = stat.mode(nearest_neighbor_classes)

    row['Class'] = predicted_class
    return row


test_data = pd.read_csv(sys.argv[1], sep=' ')
training_data = pd.read_csv(sys.argv[2], sep=' ')

X = training_data.drop("Class", axis=1)
X = X.values
y = training_data["Class"]
y = y.values

prediction_data = test_data.apply(make_prediction, axis=1)

prediction_correct = prediction_data['Class'] == training_data['Class']

proportion = prediction_correct.value_counts(normalize=True)
print(prediction_data['Class'].tolist())

print(proportion)
