import numpy as np
import scikitplot as skplt
import sklearn.linear_model as skl_lm
from sklearn.metrics import confusion_matrix, classification_report, precision_score, roc_curve, auc, log_loss
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

# Helper functions to print classification diagnostics
def print_classification_statistics(model, X_test, y_test, labels=None):
    y_pred = model.predict(X_test)
    print('Classification Report:')
    print(classification_report(y_test, y_pred, target_names=labels, digits=3))
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm, columns=pd.MultiIndex.from_product([['Predicted'],['True', 'False']]),
                         index=pd.MultiIndex.from_product([['Real'],['True', 'False']]))
    print('Confusion Matrix:')
    print(df_cm)
    print()
    
def plot_ROC(model, X_test, y_test, label=''):
    y_prob = model.predict_proba(X_test)
    ax = skplt.metrics.plot_roc(y_test, y_prob, plot_micro=False, plot_macro=False)
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > 0:
        labels[0] = labels[0].replace('ROC curve of class 1', label)
    ax.legend(handles, labels, loc="lower right")
    
def print_OLS_error_table(model, X_train, y_train):
    params = np.append(model.intercept_, model.coef_)
    predictions = model.predict(X_train)
    
    if isinstance(X_train, pd.DataFrame):
        X_cols = X_train.columns
        X_train = X_train.values
    else:
        X_cols = [f'Feature {num}' for num in range(1, len(X_train))]

    newX = pd.DataFrame({"Constant": np.ones(len(X_train))}).join(pd.DataFrame(X_train))
    
    if isinstance(model, skl_lm.LinearRegression):
        # for linear regression:
        MSE = (sum((y_train-predictions)**2))/(len(newX)-len(newX.columns))
        var_b = MSE*(np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    elif isinstance(model, skl_lm.LogisticRegression):
        # for logistic regression:
        pred_prob = model.predict_proba(X_train)
        W = np.diagflat(pred_prob[:, 1]*(1-pred_prob[:, 1]))
        cov = np.dot(newX.T, np.dot(W, newX))
        var_b = np.linalg.inv(cov).diagonal()
        
    std_errs = np.sqrt(var_b)
    t_values = params/std_errs
    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in t_values]

    std_errs = np.round(std_errs,3)
    t_values = np.round(t_values,3)
    p_values = np.round(p_values,3)
    params = np.round(params,4)
    
    # log likelihood and AIC
    y_pred = model.predict_proba(X_train)
    LLK = -log_loss(y_train, y_pred, normalize=False) 
    df_model = len(*model.coef_)
    aic = 2*(df_model+1) - 2*LLK

    model_stats = pd.DataFrame(np.array([params, std_errs, t_values, p_values]).T, 
                               index=['Intercept', *X_cols], 
                               columns=['Coefficients', 'Standard Errors', 't values', 'p values'])
    
    print(f'No. Observations: {len(y_train)}')
    print(f'Df Residuals: {len(y_train)-df_model-1}')
    print(f'Df Model: {df_model}')
    print(f'Log-Likelihood: {LLK:.2f}')
    print(f'AIC: {aic:.2f}')
    print(model_stats)
    print()
    
def plot_classification(model, X_test, y_test):   
    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    
    data1 = X_test[y_test==1]
    data0 = X_test[y_test==0]
    
    # Data
    ax.scatter(data1.values[:,0], data1.values[:,1], s=20, c='red', marker='o', label='1')
    ax.scatter(data0.values[:,0], data0.values[:,1], s=20, c='blue', marker='o', label='1')
    ax.set_xlabel(data1.columns[0]);
    ax.set_ylabel(data1.columns[1]);
    
    maxX1, maxX2 = X_test.max()
    minX1, minX2 = X_test.min()
    N_points_grid = 200
    xx, yy = np.meshgrid(np.linspace(minX1, maxX1, N_points_grid), np.linspace(minX2, maxX2, N_points_grid))
    X = np.c_[xx.ravel(), yy.ravel()]
    est_region0 = model.predict(X) > 0
    est_region1 = np.logical_not(est_region0)

    # LDA regions
    plt.contourf(xx, yy, est_region1.reshape(xx.shape), alpha=0.5, colors='red', levels=[0.5, 1.0])
    plt.contourf(xx, yy, est_region0.reshape(xx.shape), alpha=0.5, colors='blue', levels=[0.5, 1.0]);