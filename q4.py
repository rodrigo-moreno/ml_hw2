import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from data import load_superconduct

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
def dispatch(mtype, hvalue):
    """
    A function that dispatches the model with the required metaparameter value.

    Input:
    - mtype: string. Represents the type of model required.
    - hvalue: int. Represents the value of the complexity metaparameter.

    Output:
    - model: a model of the required type with the required metaparameter value
    """
    if mtype.lower() == 'knn':
        return KNeighborsRegressor(n_neighbors = hvalue)
    elif mtype.lower() == 'tree':
        return DecisionTreeRegressor(max_depth = hvalue)
    elif mtype.lower() == 'reg':
        return Ridge(alpha = hvalue)
    else:
        raise TypeError(f'Model of type {mtype} is not supported.')

def stats(model_type, values, X, y, folds = 10):
    """
    Get the statistics of the model. This does K Folds with a range of
    metaparameter values on a certain model type, and returns the parameter
    value that has the highest validation accuracy.

    Input:
    - model_type: str specifying what type of model is being crossvalidated.
      Either 'knn', 'tree' or 'reg'.
    - values: iterable containing the values for the metaparameter.
    - X, y: iterables containing the training data and their respective output.
    - folds: how many folds are going to be done. Defaults to 10.

    Output:
    - lambda_opt: value of the metaparameter with the best performance.
    """
    Xtr = X[:20000, :]
    ytr = y[:20000]
    Xte = X[20000:, :]
    yte = y[20000:]

    stats = np.zeros((len(values), 4))
    for ii, val in enumerate(values):
        model = dispatch(model_type, val)
        fit_acc = np.zeros(folds)
        pred_acc = np.zeros(folds)

        skf = KFold(n_splits = folds, shuffle = True,random_state = 0)

        for fold, (tr_idx, te_idx) in enumerate(skf.split(Xte, yte)):
            model.fit(Xtr[tr_idx], ytr[tr_idx])
            fit_acc[fold] = model.score(Xtr[tr_idx], ytr[tr_idx])
            pred_acc[fold] = model.score(Xtr[te_idx], ytr[te_idx])
        stats[ii, 0] = np.mean(fit_acc)
        stats[ii, 1] = np.std(fit_acc)
        stats[ii, 2] = np.mean(pred_acc)
        stats[ii, 3] = np.std(pred_acc)

    best = np.argmax(stats[:, 2])
    return values[best]


def variance_bias_computation(model_type,hvalue, X_train, y_train, X_test, y_test, num_rounds, random_seed):
    """
        A function that computes the expected error, variance and bias values for a certain model.

        Input:
        - model_type: str specifying what type of model is being studied
        - hvalue: value of the hyper-parameter integrated in the model
        - X_train: Training set features
        - y_train: Training set labels
        - X_test: Test set features
        - y_test: Testing set labels
        - num_rounds: Number of rounds for training
        - random_seed

        Output:
        - avg_expected_loss: average expected loss for the given model
        - avg_bias: average bias for the given model
        - avg_variance: average variance for the given model

        Todo: change the num_rounds stuff and implement KFOLD: between lines 95 and 99
        """

    np.random.seed(random_seed)

    errors = []
    all_predictions = np.zeros((num_rounds, len(y_test)))

    for i in range(num_rounds):
        # Random subset of training set (?)
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_train_subset = X_train[indices]
        y_train_subset = y_train[indices]

        model = dispatch(model_type, hvalue)

        trained_model = model.fit(X_train_subset, y_train_subset)
        predictions = trained_model.predict(X_test)

        all_predictions[i] = predictions
        errors.append(mean_squared_error(y_test, predictions))

    avg_expected_loss = np.mean(errors)
    avg_predictions = np.mean(all_predictions, axis=0)
    avg_bias = np.mean((avg_predictions - y_test) ** 2)
    avg_var = np.mean(np.var(all_predictions, axis=0))

    return avg_expected_loss, avg_bias, avg_var


def estimation(model_type, hvalue,X, y):
    """
    Compare the error of the model with the training and test samples, by
    varying the size of the training sample.

    Input:
    - model: string of model to be tested.
    - hvalue: hyperparameter value
    - X, y: LS sample. X are the attributes, y the output.

    Output:
    - Figure of the realtionship between training size and errors.

    NOTE: STILL NEED TO ADD REPETITIONS TO THIS, SO THAT THE CURVES ARE SMOOTH
    AND WE CAN ACTUALLY SEE NICE THINGS.
    """
    proportions = np.arange(10, 91, 1)
    bias_Class, var_Class, error_Class, = [], [], []
    for ii, per in enumerate(proportions):
        idx = int(np.round(len(y) * per/100))
        X_train = X[:idx, :]
        y_train = y[:idx]

        X_test = X[idx:, :]
        y_test = y[idx:]
        avg_expected_loss, avg_bias, avg_var = variance_bias_computation(model_type, hvalue, X_train, y_train, X_test,
                                                                         y_test, random_seed=123, num_rounds=100)
        bias_Class.append(avg_bias)
        var_Class.append(avg_var)
        error_Class.append(avg_expected_loss)

    fig, ax = plt.subplots()
    plt.plot(enumerate(proportions), error_Class, 'red', label='total_error', linestyle='dashed')
    plt.plot(enumerate(proportions), bias_Class, 'brown', label='bias^2')
    plt.plot(enumerate(proportions), var_Class, 'yellow', label='variance')
    plt.xlabel(f'Algorithm Complexity: {model_type}')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
    return fig



if __name__ == '__main__':
    # load dataset and distribute data.
    X, y = load_superconduct()
    model_type = 'tree'

    metaparameters = np.arange(1, 50, 2)
    best = stats(model_type, metaparameters, X, y)
    print(f'Regression tree: chose model with parameter= {best}') # Change
    estimation(model_type, best, X, y)

    model_type = 'knn'
    metaparameters = np.arange(1, 50, 2)
    best = stats(model_type, metaparameters, X, y)
    print(f'KNN: chose model with parameter = {best}')
    estimation(model_type,best, X, y)

    model_type = 'reg'
    metaparameters = np.arange(1, 50, 2)
    best = stats(model_type, metaparameters, X, y)
    print(f'Ridge regreesion: chose model with parameter = {best}')
    estimation(model_type,best, X, y)
