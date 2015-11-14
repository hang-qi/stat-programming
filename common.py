import numpy as np


# load data
def load_digits(subset=None, normalize=True):
    """
    Load digits and labels from digits.csv.

    Args:
        subset: A subset of digit from 0 to 9 to return.
                If not specified, all digits will be returned.
        normalize: Whether to normalize data values to between 0 and 1.

    Returns:
        digits: Digits data matrix of the subset specified.
                The shape is (n, p), where
                    n is the number of examples,
                    p is the dimension of features.
        labels: Labels of the digits in an (n, ) array.
                Each of label[i] is the label for data[i, :]
    """
    # load digits.csv, adopted from sklearn.
    import pandas as pd
    df = pd.read_csv('digits.csv')

    # only keep the numbers we want.
    if subset is not None:
        df = df[df.iloc[:,-1].isin(subset)]

    # convert to numpy arrays.
    digits = df.iloc[:,:-1].values.astype('float')
    labels = df.iloc[:,-1].values.astype('int')

    # Normalize digit values to 0 and 1.
    if normalize:
        digits -= digits.min()
        digits /= digits.max()

    # Change the labels to 0 and 1.
    for i in xrange(len(subset)):
        labels[labels == subset[i]] = i

    labels = labels.reshape((labels.shape[0], 1))
    return digits, labels


def split_samples(digits, labels):
    """Split the data into a training set (70%) and a testing set (30%)."""
    num_samples = digits.shape[0]
    num_training = round(num_samples * 0.7)
    indices = np.random.permutation(num_samples)
    training_idx, testing_idx = indices[:num_training], indices[num_training:]
    return (digits[training_idx], labels[training_idx],
            digits[testing_idx], labels[testing_idx])
