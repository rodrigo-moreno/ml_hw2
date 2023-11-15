import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from data import load_superconduct

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

def splitting():
    """
    A function that deals with splitting the information in the required
    subsets.
    """
    pass


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


def plot_stats(values, stats):
    """
    A function that plots the results of the k-fold cross-validation. Its main
    purpose is to declutter the stats function.

    Input:
    - values: iterable of values for the metaparameters.
    - stats: array with four columns containing the results from CV.
    """
    fig, ax = plt.subplots()
    ax.plot(values, stats[:, 0], 'b')
    ax.plot(values, stats[:, 2], 'r')
    ax.legend(['Training', 'Validation'])
    ax.fill_between(values, stats[:, 0] - stats[:, 1], stats[:, 0] + stats[:, 1],
                    alpha = 0.5)
    ax.fill_between(values, stats[:, 2] - stats[:, 3], stats[:, 2] + stats[:, 3],
                    alpha = 0.5)
    ax.set_ylim(0, 1)

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Metaparameter value')
    return fig


def stats(model_type, values, X, y, folds = 10):
    Xtr = X[:20000, :]
    ytr = y[:20000]
    Xte = X[20000:, :]
    yte = y[20000:]

    stats = np.zeros((len(values), 4))
    for ii, val in enumerate(values):
        model = dispatch(model_type, val)
        fit_acc = np.zeros(folds)
        pred_acc = np.zeros(folds)

        skf = KFold(n_splits = folds, shuffle = True,
                              random_state = 0)
        for fold, (tr_idx, te_idx) in enumerate(skf.split(Xte, yte)):
            model.fit(Xtr[tr_idx], ytr[tr_idx])
            fit_acc[fold] = model.score(Xtr[tr_idx], ytr[tr_idx])
            pred_acc[fold] = model.score(Xtr[te_idx], ytr[te_idx])
        stats[ii, 0] = np.mean(fit_acc)
        stats[ii, 1] = np.std(fit_acc)
        stats[ii, 2] = np.mean(pred_acc)
        stats[ii, 3] = np.std(pred_acc)

    fig = plot_stats(values, stats)
    fig.suptitle(f'Accuracy of {model_type} model over CV with {folds} folds')
    plt.savefig(f'{model_type}_{folds}folds_{min(values)}to{max(values)}.pdf',
                format = 'pdf')
    #plt.show()
    best = np.argmax(stats[:, 2])
    return values[best]


def estimation(model, X, y):
    proportions = np.arange(10, 91, 1)
    errors = np.zeros((len(proportions), 2))
    for ii, per in enumerate(proportions):
        idx = int(np.round(len(y) * per/100))
        Xtr = X[:idx, :]
        ytr = y[:idx]
        Xte = X[idx:, :]
        yte = y[idx:]

        model.fit(Xtr, ytr)
        errors[ii, 0] = 1 - model.score(Xtr, ytr)
        errors[ii, 1] = 1 - model.score(Xte, yte)

    fig, ax = plt.subplots()
    ax.plot(proportions, errors[:, 0], 'b')
    ax.plot(proportions, errors[:, 1], 'r')
    ax.legend(['Training', 'Test'])
    ax.set_ylabel('Error')
    ax.set_xlabel('Percentual size of training sample (%)')
    #plt.show()
    return fig


if __name__ == '__main__':
    X, y = load_superconduct()

    model_type = 'tree'
    metaparameters = np.arange(1, 50, 2)
    best = stats(model_type, metaparameters, X, y)
    #print(f'Chose model with parameter = {best}')
    estimation(dispatch(model_type, best), X, y)
    plt.savefig(f'{model_type}_estimation.pdf', format = 'pdf')

    model_type = 'reg'
    metaparameters = np.arange(1, 50, 2)
    best = stats(model_type, metaparameters, X, y)
    #print(f'Chose model with parameter = {best}')
    estimation(dispatch(model_type, best), X, y)
    plt.savefig(f'{model_type}_estimation.pdf', format = 'pdf')

    model_type = 'knn'
    metaparameters = np.arange(1, 50, 2)
    best = stats(model_type, metaparameters, X, y)
    #print(f'Chose model with parameter = {best}')
    estimation(dispatch(model_type, best), X, y)
    plt.savefig(f'{model_type}_estimation.pdf', format = 'pdf')

