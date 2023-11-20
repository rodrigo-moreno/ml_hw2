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


def variance_bias_computation_kfold(model_type, hvalue, X_train, y_train,
                                    X_test, y_test, random_seed, n_splits):
    """
    A function that computes the expected error, variance and bias values
    for a certain model.

    Input:
    - model_type: str specifying what type of model is being studied
    - hvalue: value of the hyper-parameter integrated in the model
    - X_train: Training set features
    - y_train: Training set labels
    - X_test: Test set features
    - y_test: Testing set labels
    - random_seed
    - n_splits: Number of splits of kfold

    Output:
    - avg_expected_loss: average expected loss for the given model
    - avg_bias: average bias for the given model
    - avg_variance: average variance for the given model
    """

    np.random.seed(random_seed)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    errors = np.zeros(n_splits)
    all_predictions = np.zeros((n_splits, len(y_test)))

    for fold_idx, (train_index, val_index) in enumerate(kfold.split(X_train)):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        model = dispatch(model_type, hvalue)

        trained_model = model.fit(X_train_fold, y_train_fold)
        predictions = trained_model.predict(X_test)

        all_predictions[fold_idx, :] = predictions
        errors[fold_idx] = mean_squared_error(y_test, predictions)
        #print(predictions[:5])
        #print(y_test[:5])
        #print(y_test[:5] - predictions[:5])
        #print(errors[fold_idx])

    # Calculate average expected loss, bias, and variance
    avg_expected_loss = np.mean(errors)
    avg_predictions = np.mean(all_predictions, axis=0)
    avg_bias = np.mean((avg_predictions - y_test) ** 2)
    avg_var = np.mean(np.var(all_predictions, axis=0))

    return avg_expected_loss, avg_bias, avg_var


def models_assessment(model_type, hvalues, X_train, y_train, X_test, y_test):
    """
    Parameters
    ----------
    model_type: string. Represents the type of model required.
    hvalues: array containing hyperparameter values
    X_train: array containing training samples
    y_train: array containing training labels
    X_test: array containing test samples
    y_test: array containing test labels
    """
    #bias_Class, var_Class, error_Class, = [], [], []

    bias_Class = np.zeros(len(hvalues))
    var_Class = np.zeros(len(hvalues))
    error_Class = np.zeros(len(hvalues))
    for ii, kk in enumerate(hvalues):#range(len(hvalue)):
        avg_expected_loss, avg_bias, avg_var = variance_bias_computation_kfold(model_type, kk, X_train, y_train,
                                                                       X_test ,y_test, random_seed=123, n_splits=10)
        bias_Class[ii] = avg_bias
        var_Class[ii] = avg_var
        error_Class[ii] = avg_expected_loss
        #print(f"Average expected loss with model {model_type} and hyperparameter {hvalue[k]}: {avg_expected_loss}")
        #print(f"Average bias with model {model_type} and hyperparameter {hvalue[k]}: {avg_bias}")
        #print(f"Average variance with model {model_type} and hyperparameter {hvalue[k]}: {avg_var}")

    fig, ax = plt.subplots()
    ax.plot(hvalues, bias_Class, 'brown', label='bias^2')
    ax.plot(hvalues, var_Class, 'yellow', label='variance')
    ax.plot(hvalues, error_Class, 'red', label='total_error', linestyle='dashed')
    ax.set_xlabel(f'Algorithm Complexity: {model_type}')
    ax.set_ylabel('Error')
    ax.legend()
    plt.savefig(f'{model_type}_{len(y_train)+len(y_test)}_assessment.pdf', format = 'pdf')
    #plt.show()
    return


if __name__ == '__main__':
    # load dataset and distribute data.
    X, y = load_superconduct()
    number_of_ls = 500
    X_train = X[0:int(np.ceil(number_of_ls*0.7)), :]
    y_train = y[0:int(np.ceil(number_of_ls*0.7))]

    X_test = X[int(np.ceil(number_of_ls*0.7)):number_of_ls, :]
    y_test = y[int(np.ceil(number_of_ls*0.7)):number_of_ls]

    # assesment of Decision Tree
    model_type = 'tree'
    metaparameters = np.arange(1, 50, 2)
    models_assessment(model_type, metaparameters, X_train, y_train, X_test, y_test)
    print(f'Finished {model_type}')

    # assesment of Ridge Regression
    model_type = 'reg'
    metaparameters = np.arange(0, 11, 2)
    models_assessment(model_type, metaparameters, X_train, y_train, X_test, y_test)
    print(f'Finished {model_type}')

    # assesment of KNN
    model_type = 'knn'
    metaparameters = np.arange(1, 11, 1)
    models_assessment(model_type, metaparameters, X_train, y_train, X_test, y_test)
    print(f'Finished {model_type}')
