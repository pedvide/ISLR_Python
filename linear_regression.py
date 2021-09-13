import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from statsmodels.graphics.gofplots import ProbPlot

def plot_lm(model_fit, data, key, n=3):
    
    # fitted values (need a constant term for intercept)
    model_fitted_y = model_fit.fittedvalues
    # model residuals
    model_residuals = model_fit.resid
    # normalized residuals
    model_norm_residuals = model_fit.get_influence().resid_studentized_internal
    # absolute squared normalized residuals
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    # absolute residuals
    model_abs_resid = np.abs(model_residuals)
    # leverage, from statsmodels internals
    model_leverage = model_fit.get_influence().hat_matrix_diag
    # cook's distance, from statsmodels internals
    model_cooks = model_fit.get_influence().cooks_distance[0]
    
    fig = plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(12)
    
    # first plot
    plot_lm_1 = plt.subplot(2, 2, 1)

    temp_plot = sns.residplot(x=model_fitted_y, y=key, data=data, 
                              lowess=True, 
                              scatter_kws={'alpha': 0.5}, 
                              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    plot_lm_1.set_title('Residuals vs Fitted')
    plot_lm_1.set_xlabel('Fitted values')
    plot_lm_1.set_ylabel('Residuals')

    # annotations
    abs_resid = model_abs_resid.sort_values(ascending=False)
    abs_resid_top_3 = abs_resid[:3]

    for i in abs_resid_top_3.index:
        plot_lm_1.annotate(i, xy=(model_fitted_y[i], model_residuals[i]));
        
    # second plot
    plot_lm_2 = plt.subplot(2, 2, 2)
    QQ = ProbPlot(model_norm_residuals)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        temp_plot = QQ.qqplot(line='45', alpha=0.5, color='lightblue', lw=1, ax=plot_lm_2)

    plot_lm_2.set_title('Normal Q-Q')
    plot_lm_2.set_xlabel('Theoretical Quantiles')
    plot_lm_2.set_ylabel('Standardized Residuals');

    # annotations
    abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
    abs_norm_resid_top_3 = abs_norm_resid[:3]

    for r, i in enumerate(abs_norm_resid_top_3):
        plot_lm_2.annotate(i, xy=(np.flip(QQ.theoretical_quantiles, 0)[r],model_norm_residuals[i]));
        
    # third plot
    plot_lm_3 = plt.subplot(2, 2, 3)

    plt.scatter(x=model_fitted_y, y=model_norm_residuals_abs_sqrt, alpha=0.5)
    sns.regplot(x=model_fitted_y, y=model_norm_residuals_abs_sqrt, 
                scatter=False, 
                ci=False, 
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    plot_lm_3.set_title('Scale-Location')
    plot_lm_3.set_xlabel('Fitted values')
    plot_lm_3.set_ylabel('$\sqrt{|Standardized Residuals|}$');

    # annotations
    abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
    abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]

    for i in abs_norm_resid_top_3:
        try:
            plot_lm_3.annotate(i, xy=(model_fitted_y[i], model_norm_residuals_abs_sqrt[i]));
        except KeyError as err:
            continue
        
    # fourth plot
    plot_lm_4 = plt.subplot(2, 2, 4)

    plt.scatter(x=model_leverage, y=model_norm_residuals, alpha=0.5)
    sns.regplot(x=model_leverage, y=model_norm_residuals, 
                scatter=False, 
                ci=False, 
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    plot_lm_4.set_xlim(0, 0.20)
    plot_lm_4.set_ylim(-3, 5)
    plot_lm_4.set_title('Residuals vs Leverage')
    plot_lm_4.set_xlabel('Leverage')
    plot_lm_4.set_ylabel('Standardized Residuals')

    # annotations
    leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]

    for i in leverage_top_3:
        plot_lm_4.annotate(i, xy=(model_leverage[i], model_norm_residuals[i]))

    # shenanigans for cook's distance contours
    def graph(formula, x_range, label=None):
        x = x_range
        y = formula(x)
        plt.plot(x, y, label=label, lw=1, ls='--', color='red')

    p = len(model_fit.params) # number of model parameters

    graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), 
          np.linspace(0.001, 0.200, 50), 
          'Cook\'s distance') # 0.5 line
    graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), 
          np.linspace(0.001, 0.200, 50)) # 1 line
    plt.legend(loc='upper right');
    plt.xlim(xmax=np.max(model_cooks))
    
    plt.tight_layout()