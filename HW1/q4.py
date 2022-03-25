import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import random

str_to_int = {}
int_to_str = {}

# load data
df = pd.read_csv('iris.data', header=None)
classes = df[df.columns[-1]].unique()
for i, c in enumerate(classes):
    str_to_int[c] = i
    int_to_str[i] = c

df.replace(str_to_int, inplace=True)
dataset = df.to_numpy()

dataset_dict = {}
for k in int_to_str.keys():
    dataset_dict[k] = []

for row in dataset:
    dataset_dict[row[-1]].append(row[:-1])

for k in int_to_str.keys():
    dataset_dict[k] = np.array(dataset_dict[k])

"""split train and test data 0.1 of whole data will use for"""
def split_train_test():
    test_num = 0.1 * len(df)
    each_class_test_number = test_num / 3
    test_dataset = {}
    for c in int_to_str.keys():
        len_ci = len(dataset_dict[c])
        random_indexs = random.sample(
            range(int(len_ci)), int(each_class_test_number))
        test_ci = dataset_dict[c][random_indexs]
        dataset_dict[c] = np.delete(dataset_dict[c], random_indexs, axis=0)
        print(dataset_dict[c].shape)
        test_dataset[c] = test_ci
    return test_dataset


# compute mean of a class
""" return type is 1-D array"""
def compute_mean(values):
    return values.mean(axis=0)


# compute std of a class
""" return type is 1-D arrray"""
def compute_stdev(values, means):
    population_size, _ = values.shape
    variance = np.sum((values - means)**2, axis=0) / float(population_size-1)
    return np.sqrt(variance)


""" probability of each class in general """
def probility_class():
    keys = int_to_str.keys()
    total = dataset.shape[0] - 0.1 * dataset.shape[0]
    probs = []
    for key in keys:
        probs.append(len(dataset_dict[key]) / total)
    return np.array(probs)

""" compute gussian probability """
def compute_probability_per_class(class_prob, x, means, stdevs):
    return ((1/np.sqrt(2*np.pi*stdevs**2))*np.exp(-0.5 * ((x-means)/stdevs)**2)).prod()*(class_prob)

""" compute standard devation and mean and probability of each class and features"""
def compute_class_details():
    class_probs = probility_class()
    class_stdevs = []
    class_means = []
    for cl in int_to_str.keys():
        mean = compute_mean(np.array(dataset_dict[cl]))
        class_means.append(mean)
        class_stdevs.append(compute_stdev(np.array(dataset_dict[cl]), mean))

    class_stdevs = np.array(class_stdevs)
    class_means = np.array(class_means)
    return (class_stdevs, class_means, class_probs)


""" predict probability per each class and choose the higher one"""
def predict(X_test, class_stdevs, class_means, class_probs):
    best_label, best_prob = None, -1
    for cl in int_to_str.keys():
        prob = compute_probability_per_class(
            class_prob=class_probs[cl], x=X_test, means=class_means[cl], stdevs=class_stdevs[cl])
        if best_prob < prob:
            best_prob = prob
            best_label = cl
    return best_label

""" Gussian naive bayes"""
def main():
    test_dataset = split_train_test()
    class_stdevs, class_means, class_probs = compute_class_details()
    accuracy = []

    for key in test_dataset.keys():
        for x_test in test_dataset[key]:
            probability = predict(x_test, class_stdevs,
                                  class_means, class_probs)
            accuracy.append(key == probability)
    acc = sum(accuracy)/len(accuracy) * 100
    
    print(f'The accuracy is {np.round(acc, 2)}%')


if __name__ == '__main__':
    main()