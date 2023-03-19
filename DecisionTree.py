import pandas as pd
import sys
from DecisionTreeNode import DecisionTreeNode as Node
from random import randint


def type_and_probability(instances):
    proportions = instances['Class'].value_counts(normalize=True)
    if 'live' not in proportions.index:
        return 'die', 1
    elif 'die' not in proportions.index:
        return 'live', 1

    if proportions['live'] < proportions['die']:
        return 'die', proportions['die']
    elif proportions['live'] > proportions['die']:
        return 'live', proportions['live']
    else:
        i = randint(0, 1)
        if i == 0:
            return 'die', proportions['die']
        else:
            return 'live', proportions['live']


def purity(instances):  # assumes only two possible types
    amounts = instances['Class'].value_counts()
    if amounts.size < 2:
        return 0

    return (amounts[0] * amounts[1]) / (amounts[0] + amounts[1]) ** 2


def weighted_average(n, n1, n2, p1, p2):
    return ((n1 / n) * p1) + ((n2 / n) * p2)


def build_tree(attributes, instances):
    if instances.size == 0:
        name, probability = type_and_probability(training_data)
        return Node(name=name, probability=probability, size=0)

    p = purity(instances)
    if p == 0:
        name = instances['Class'].reset_index(drop=True)[0]
        temp = len(instances.index)
        node = Node(name=name, probability=1, size=len(instances.index))
        return Node(name=name, probability=1, size=len(instances.index))
    elif attributes.size == 0:
        name, probability = type_and_probability(instances)
        return Node(name=name, probability=probability, size=instances.size)
    else:
        best_avg = 1
        best_attr = None
        best_true_set = None
        best_false_set = None

        for attribute in attributes:
            true_set = instances.loc[instances[attribute] == True]
            false_set = instances.loc[instances[attribute] == False]
            purity_true = purity(true_set)
            purity_false = purity(false_set)
            avg = weighted_average(instances.size, true_set.size, false_set.size, purity_true, purity_false)

            if avg < best_avg:
                best_avg = avg
                best_attr = attribute
                best_true_set = true_set
                best_false_set = false_set

        left = build_tree(attributes.drop(best_attr), best_true_set)
        right = build_tree(attributes.drop(best_attr), best_false_set)
        return Node(attribute=best_attr, left_child=left, right_child=right)


def make_prediction(row, node):
    if node.probability is not None:
        row['prediction'] = node.name
        return row
    attribute = node.attribute
    if row[attribute]:
        return make_prediction(row, node.left_child)
    else:
        return make_prediction(row, node.right_child)


# use pandas to read in data
test_data = pd.read_csv(sys.argv[1], sep=' ')
training_data = pd.read_csv(sys.argv[2], sep=' ')

attributes = training_data.columns
attributes = attributes.drop('Class')

node = build_tree(attributes, training_data)

print(node.print_node('  '))  # need to fix this

test_data['prediction'] = None
test_data = test_data.apply(lambda row: make_prediction(row, node), axis=1)

test_data.to_excel('output decision tree.xlsx')

correct_predictions = test_data['Class'] == test_data['prediction']
proportion = correct_predictions.value_counts(normalize=True)
print(proportion)
