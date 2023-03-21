import pandas as pd
import sys
from DecisionTreeNode import DecisionTreeNode as Node
from random import randint


def type_and_probability(instances):  # assumes only two possible types
    proportions = instances['Class'].value_counts(normalize=True)

    if proportions.size < 2:
        return 1

    if proportions[0] < proportions[1]:
        return proportions.index[1], proportions[1]
    elif proportions[0] > proportions[1]:
        return proportions.index[0], proportions[0]
    else:
        i = randint(0, 1)
        if i == 0:
            return proportions.index[1], proportions[1]
        else:
            return proportions.index[0], proportions[0]


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

    if purity(instances) == 0:
        name = instances['Class'].reset_index(drop=True)[0]
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

print(node.print_node(1))

test_data['prediction'] = None
test_data = test_data.apply(lambda row: make_prediction(row, node), axis=1)

test_data.to_excel('output decision tree.xlsx')

correct_predictions = test_data['Class'] == test_data['prediction']
proportion = correct_predictions.value_counts(normalize=True)
print(proportion)
