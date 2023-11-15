from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# You can use this function in a script (located in the same folder) like this:
# 
# from data import load_superconduct
#
# X, y = load_superconduct()

def load_superconduct():
    """Loads and returns the (normalized) Superconduct dataset from OpenML.

    Return
    ------
    X : array of shape (21263, 79)
    The feature matrix (input).
    y : array of shape (21263,)
    The output values vector.
    """
    dataset = fetch_openml(data_id=44148, parser='auto')

    X, y = dataset.data, dataset.target
    X, y = X.to_numpy(), y.to_numpy()

    # Normalization is important for ridge regression and k-NN.
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Shuffle the data
    X, y = shuffle(X, y, random_state=42)

    return X, y

