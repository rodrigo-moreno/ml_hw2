import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

from data import load_superconduct

from q2 import dispatch

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


def bootstrap_statistics(model_type, hvalue, N, Xtr, ytr, Xte, yte):
    """
    Makes a prediction of the data Xte using N different models and bootstrap
    sampling.

    Input:
    - model_type: str. Determines the type of model to be used.
    - hvalue: int. Value of metaparameter of the model.
    - N: int. Amount of models to be used to make the prediction.
    - Xtr: iterable of attributes for model training.
    - ytr: iterable of output features for model training.
    - Xte: iterable of attributes for model testing.
    - yte: iterable of output features for model testing.

    Output:
    - pred: iterable of predictions for each element in Xte.
    """
    pred = np.zeros((N, len(yte)))
    for ii in range(N):
        #print(f'On round {ii} for case of {N}.')
        model = dispatch(model_type, hvalue)
        Xr, yr = resample(Xtr, ytr)#, random_state = ii)
        model.fit(Xr, yr)
        pred[ii, :] = model.predict(Xte)
    pred = np.mean(pred, axis = 0)
    return pred


def foo(model_type, hvalue, models, X, y, repetitions = 10):
    """
    Calculates the bias, error and var of prediction with a certain model type,
    using bootstrap sampling.

    Input:
    - model_type: str. Indicates the type of model to be used.
    - hvalue: int. Indicates metaparameter value for the model.
    - models: iterative containing the amount of models to be trained and used
      for prediction.
    - X: iterable of attributes to be split into learning and testing samples.
    - y: iterable of output features to be split into learning and testing
      samples.
    - repetitions: int. The amount of iterations to do. Mostly for statistical
      purposes.

    Output:
    - variance, bias and error of the ``reptitions'' predictions for each
      instance or ``models''.
    """
    Xtr = X[:2000, :]
    ytr = y[:2000]
    Xte = X[2000:, :]
    yte = y[2000:]

    varis = np.zeros(len(models))
    biases = np.zeros(len(models))
    errors = np.zeros(len(models))

    for ii, amount in enumerate(models):
        print(f'Working on model with {amount} predictors.')
        preds = np.zeros((repetitions, len(yte)))
        error = np.zeros(repetitions)
        for rep in range(repetitions):
            preds[rep, :] = bootstrap_statistics(model_type, hvalue,
                                                 amount, Xtr, ytr, Xte, yte)
            error[rep] = mean_squared_error(yte, preds[rep, :])
            print(f'Round {rep}, error {error[rep]}')
        varis[ii] = np.mean(np.var(preds, axis = 0))
        mean_pred = np.mean(preds, axis = 0)
        biases[ii] = np.mean((mean_pred - yte) ** 2)
        errors[ii] = np.mean(error)
        print(errors[ii])

    return varis, biases, errors


def plot_boot(models, std, bias, error):
    fig, ax = plt.subplots()
    ax.plot(models, std, 'b')
    ax.plot(models, bias, 'r')
    ax.plot(models, error, 'k')
    ax.plot(models, std + bias, 'r:')
    ax.legend([r'$\sigma^2$', r'bias$^2$', r'Error', r'$\sigma^2 + bias^2$'],
              fontsize = 24)
    ax.set_xlabel('Number of models used for prediction', fontsize = 24)
    return fig


if __name__ == '__main__':
    X, y = load_superconduct()

    model_type = 'knn'
    hvalue = 2
    models = np.arange(1, 61, 10)
    #models = [1, 5, 10]
    std, bias, error = foo(model_type, hvalue, models, X, y)
    f = plot_boot(models, std, bias, error)
    plt.savefig(f'{model_type}_boot.pdf', format = 'pdf')
    plt.savefig(f'{model_type}_boot.png', format = 'png')

    model_type = 'reg'
    hvalue = 2
    models = np.arange(1, 61, 10)
    #models = [1, 5, 10]
    std, bias, error = foo(model_type, hvalue, models, X, y)
    f = plot_boot(models, std, bias, error)
    plt.savefig(f'{model_type}_boot.pdf', format = 'pdf')
    plt.savefig(f'{model_type}_boot.png', format = 'png')

    model_type = 'tree'
    hvalue = None
    models = np.arange(1, 61, 10)
    #models = [1, 5, 10]
    std, bias, error = foo(model_type, hvalue, models, X, y)
    f = plot_boot(models, std, bias, error)
    plt.savefig(f'{model_type}_boot.pdf', format = 'pdf')
    plt.savefig(f'{model_type}_boot.png', format = 'png')

