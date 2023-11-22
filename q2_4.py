import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error

from data import load_superconduct

import matplotlib.pyplot as plt
import seaborn as sns
import q2_3

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
        model = q2_3.dispatch(model_type, val)
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
    proportions = np.arange(30, 91, 10)
    bias_Class = np.zeros(len(proportions))
    var_Class = np.zeros(len(proportions))
    error_Class = np.zeros(len(proportions))
    for ii, per in enumerate(proportions):
        Xu, Xd, yu, yd = train_test_split(X, y, test_size = 1 - per / 100)
        print(f'Size of used: {len(yu)}')
        Xtr, Xte, ytr, yte = train_test_split(Xu, yu, test_size = 0.33)
        print(f'Size of train: {len(ytr)} => {len(ytr) / len(yu)}')
        print(f'Size of test: {len(yte)} => {len(yte) / len(yu)}')
        avg_expected_loss, avg_bias, avg_var = q2_3.variance_bias_computation_kfold(model_type, hvalue, Xtr, ytr, Xte,
                                                                         yte, random_seed=123, n_splits=10)
        bias_Class[ii] = avg_bias
        var_Class[ii] = avg_var
        error_Class[ii] = avg_expected_loss

    print(f'Bias: {bias_Class}')
    print(f'Var: {var_Class}')
    print(f'Error: {error_Class}')

    fig, ax = plt.subplots()
    ax.plot(proportions, var_Class, 'b')
    ax.plot(proportions, bias_Class, 'r')
    ax.plot(proportions, error_Class, 'k')
    ax.plot(proportions, var_Class + bias_Class, 'r:')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(-5, ymax)
    ax.legend([r'$\sigma^2$', r'bias$^2$', r'Error', r'$\sigma^2 + bias^2$'],
              fontsize = 24)
    ax.set_xlabel(r'Proportion of $LS$ used for training (%)',
                  fontsize = 24)
    #plt.show()
    return fig


if __name__ == '__main__':
    # load dataset and distribute data.
    X, y = load_superconduct()
    
    model_type = 'tree'
    metaparameters = np.arange(1, 50, 2)
    best = stats(model_type, metaparameters, X, y)
    print(f'Regression tree: chose model with parameter = {best}') # Change
    estimation(model_type, best, X, y)
    plt.savefig(f'{model_type}_{best}.pdf', format = 'pdf')
    plt.savefig(f'{model_type}_{best}.png', format = 'png')
    
    print('Starting fully grown trees')
    estimation(model_type, None, X, y)
    plt.savefig(f'{model_type}.pdf', format = 'pdf')
    plt.savefig(f'{model_type}.png', format = 'png')

    model_type = 'knn'
    metaparameters = np.arange(1, 50, 2)
    best = stats(model_type, metaparameters, X, y)
    print(f'KNN: chose model with parameter = {best}')
    estimation(model_type,best, X, y)
    plt.savefig(f'{model_type}.pdf', format = 'pdf')
    plt.savefig(f'{model_type}.png', format = 'png')

    model_type = 'reg'
    metaparameters = np.arange(1, 50, 2)
    best = stats(model_type, metaparameters, X, y)
    print(f'Ridge regreesion: chose model with parameter = {best}')
    estimation(model_type,best, X, y)
    plt.savefig(f'{model_type}.pdf', format = 'pdf')
    plt.savefig(f'{model_type}.png', format = 'png')

