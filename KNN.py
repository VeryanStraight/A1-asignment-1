import pandas as pd
import numpy as np
import sys
import statistics as stat


def knn(df_test, df_train, k, print_flag):
    df_train_without_class = df_train.drop("Class", axis=1)
    df_train_class = df_train["Class"]

    # convert to 2d arrays
    train_without_class = df_train_without_class.values
    train_class = df_train_class.values

    # gets the range of the data for each column
    range_data = df_train_without_class.apply(lambda c: c.max() - c.min(), axis=0).values

    # apply knn
    prediction = df_test.apply(lambda r: make_prediction(r, train_without_class, train_class, range_data, k), axis=1)

    # output
    prediction_correct = prediction['Class'] == df_test['Class']
    proportion = prediction_correct.value_counts(normalize=True)

    prediction['Original class'] = df_test['Class']
    if print_flag:
        prediction.to_excel('output knn.xlsx')
        print(prediction['Class'].values)
        print(proportion)

    return proportion


def make_prediction(row, X, y, r, k):
    # drop the actual class in the test data
    data_row = row.drop(labels=['Class'])

    # crate an array containing the difference for each row in the training data
    difference_from_training = np.linalg.norm((X - data_row.values) / r, axis=1)

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

knn(test_data, training_data, 3, True)
