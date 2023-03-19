import pandas as pd
from KNN import knn
import sys

k = 5

# use pandas to read in data
test_data = pd.read_csv(sys.argv[1], sep=' ')
training_data = pd.read_csv(sys.argv[2], sep=' ')

data = pd.concat([test_data, training_data]).reset_index(drop=True)

# split the data in to 5 equal parts using random sampling
remainder = data.copy()
data_split = []
for x in range(5):
    df = remainder.sample(frac=1 / (5 - x))
    remainder.drop(df.index, inplace=True)
    data_split.append(df.copy())
print('')
print('k-fold -----------------------')
# do the k fold
total = 0
for x in range(5):
    df_test = data_split[x]
    df_train = data.drop(df_test.index)

    proportion = knn(df_test, df_train, 3, False)
    print(proportion[0])
    total += proportion[0]

print('')
print('average: ' + str(total / 5))
