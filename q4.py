import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from data import load_superconduct

import matplotlib.pyplot as plt
import seaborn as sns
import q2

sns.set_theme()

np.random.seed(1)

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
        model = q2.dispatch(model_type, val)
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

def estimation(model_type, hvalue, X, y):
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
        avg_expected_loss, avg_bias, avg_var = q2.variance_bias_computation_kfold(model_type, hvalue, X_train, y_train, X_test,
                                                                         y_test, random_seed=123, n_splits=10)
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
