import numpy as np
# import scikitplot as skplt
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
    num_class = len(cm)
    df_cm = pd.DataFrame(cm, columns=pd.MultiIndex.from_product([['Predicted'], np.arange(0, num_class)]),
                         index=pd.MultiIndex.from_product([['Real'], np.arange(0, num_class)]))
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
    
    if isinstance(X_test, pd.DataFrame):
        X_cols = X_test.columns
        X_test = X_test.values
    else:
        X_cols = ['', '']
        
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.values
        
    # Data
    ax.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlabel(X_cols[0]);
    ax.set_ylabel(X_cols[1]);
    
    maxX1, maxX2 = np.max(X_test, axis=0)
    minX1, minX2 = np.min(X_test, axis=0)
    N_points_grid = 200
    xx, yy = np.meshgrid(np.linspace(minX1, maxX1, N_points_grid), np.linspace(minX2, maxX2, N_points_grid))
    X = np.c_[xx.ravel(), yy.ravel()]
    est_region = model.predict(X)

    # regions
    plt.contourf(xx, yy, est_region.reshape(xx.shape), cmap=plt.cm.coolwarm, alpha=0.5);
    
    return ax