import numpy as np
import pandas as pd
import sys

n = 0.05
class_map = {'b': 0, 'g': 1}


def classify(row):
    global weights
    temp = row.drop('class')
    temp = temp.drop('predicted')
    df = pd.concat({'x': temp, 'w': weights}, axis=1)

    # find predicted output
    total = sum(df.apply(lambda r: r['x'] * r['w'], axis=1))

    result = 'b'
    if total > 0:
        result = 'g'

    row['predicted'] = result
    return row


def update_weights(row):
    global weights
    temp = row.drop('class')
    df = pd.concat({'x': temp, 'w': weights}, axis=1)

    # find predicted output
    total = sum(df.apply(lambda r: r['x'] * r['w'], axis=1))

    result = 'b'
    if total > 0:
        result = 'g'

    # return if as expected
    if row['class'] is result:
        return True

    # update the weights
    weights = df.apply(lambda r: r['w'] + n * (class_map[row['class']] - class_map[result]) * r['x'], axis=1)
    return False


data = pd.read_csv(sys.argv[1], sep=' ')

weights = pd.Series(data=0, index=data.columns)
weights.drop(['class'], inplace=True)
weights['b'] = 0
data['b'] = 1

flag = True
count = 0
iterations = 0
amount_correct = 0

best_weights = weights.copy()
best_percent = 0


while flag:
    df = data.apply(update_weights, axis=1)
    iterations += 1
    amount_correct = df.value_counts(normalize=True)[0]
    if amount_correct > best_percent:
        count = 0
        best_percent = amount_correct
        best_weights = weights.copy()
    else:
        count += 1
        if count == 100:
            print('training complete - stopped converging')
            flag = False

    if df.all():
        print('training complete - all correct')
        flag = False

data['predicted'] = None
predictions = data.apply(classify, axis=1)

print("number of iterations " + str(iterations))
print("number of correct classes "+str(best_percent*len(data.index)))
print('proportion correct '+str(best_percent))
print("weights:")
print(best_weights)

predictions.drop('b', inplace=True, axis=1)
predictions.to_excel('perceptron output.xlsx')


