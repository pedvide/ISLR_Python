import warnings

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot

from scipy.interpolate import UnivariateSpline

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


def add_margins(ax, x=0.05, y=0.05):
    # This will, by default, add 5% to the x and y margins. You 
    # can customise this using the x and y arguments when you call it.

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xmargin = (xlim[1]-xlim[0])*x
    ymargin = (ylim[1]-ylim[0])*y

    ax.set_xlim(xlim[0]-xmargin, xlim[1]+xmargin)
    ax.set_ylim(ylim[0]-ymargin, ylim[1]+ymargin)
    

def residuals_vs_fitted(model, ax, show_quantiles=False, studentized=False):
    if studentized:
        residuals = model.get_influence().resid_studentized_internal
    else:
        residuals = model.resid
    abs_resid = np.abs(residuals)
    
    data = pd.DataFrame({"y_fitted": model.fittedvalues, "resid": residuals, "abs_resid": abs_resid})
    
    sns.residplot(
        data=data,
        x="y_fitted", y="resid",
        lowess=True, 
        scatter_kws={'alpha': 0.5}, 
        line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
        ax=ax
    )
    if studentized:
        ax.axhline(y=2, linestyle="--", color="grey")
        ax.axhline(y=-2, linestyle="--", color="grey")

    ax.set_title('Residuals vs Fitted')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals')

    # annotations
    data = data.sort_values("abs_resid", ascending=False)
    for i, row in data[:3].iterrows():
        ax.annotate(i, xy=(row.y_fitted, row.resid));
        
    if show_quantiles:
        data["y_fitted_bins"] = pd.cut(data.y_fitted, bins=10)
        def quantiles(df):
            d = {}
            d['resid_q90'] = df.resid.quantile(0.9)
            d['resid_q10'] = df.resid.quantile(0.1)
            return pd.Series(d, index=['resid_q10', 'resid_q90'])
        data = (
            data.set_index("y_fitted_bins")
            .join(data.groupby("y_fitted_bins").apply(quantiles)).sort_values("y_fitted")
        )
        
        xs = np.linspace(data.y_fitted.min(), data.y_fitted.max(), 1000)
        q_90 = UnivariateSpline(x=data.y_fitted, y=data.resid_q90, k=3)(xs)
        q_10 = UnivariateSpline(x=data.y_fitted, y=data.resid_q10, k=3)(xs)
        sns.lineplot(x=xs, y=q_10, color="blue", label="Quantiles 10-90", ax=ax)
        sns.lineplot(x=xs, y=q_90, color="blue", ax=ax)
        ax.legend()
        
    add_margins(ax)
    sns.despine()


def residuals_autocorrelation(model, ax):
    sm.graphics.tsa.plot_pacf(
        model.resid, lags=20, ax=ax,
        title="Partial autocorrelation of residuals", zero=False
    )
    sns.despine()


def residuals_normality(model, ax):
    std_resid = model.get_influence().resid_studentized_internal
    QQ = ProbPlot(std_resid)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        QQ.qqplot(line='45', alpha=0.5, color='lightblue', lw=1, ax=ax)

    ax.set_title('Normal Q-Q')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Standardized Residuals');

    # annotations
    abs_std_resid = np.flip(np.argsort(np.abs(std_resid)), 0)
    abs_std_resid_top_3 = abs_std_resid[:3]

    for r, i in enumerate(abs_std_resid_top_3):
        ax.annotate(i, xy=(np.flip(QQ.theoretical_quantiles, 0)[r], std_resid[i]));
        

def leverage(model, ax):
    
    std_residuals = model.get_influence().resid_studentized_internal
    leverage = model.get_influence().hat_matrix_diag
    # cook's distance, from statsmodels internals
    cooks_distance = model.get_influence().cooks_distance[0]
    
#     plt.scatter(x=leverage, y=std_residuals, alpha=0.5)
    sns.regplot(
        x=leverage, y=std_residuals, 
        scatter=True, 
        ci=False, 
        lowess=True,
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
        ax=ax
    )

    ax.set_xlim(leverage.min()*0.9, leverage.max())
    ax.set_ylim(std_residuals.min(), std_residuals.max())
    ax.set_title('Residuals vs Leverage')
    ax.set_xlabel('Leverage')
    ax.set_ylabel('Standardized Residuals')

    # annotations
    leverage_top_3 = np.flip(np.argsort(cooks_distance), 0)[:3]

    for i in leverage_top_3:
        ax.annotate(i, xy=(leverage[i], std_residuals[i]))

    df_model = model.df_model
    n = len(leverage)
    cooks_threshold = 4/(n - df_model - 1)
    leverage_threshold = 4 * (df_model)/n
    
    ax.axvline(
        x=leverage_threshold, 
        label=f"Leverage threshold ({leverage_threshold:.3f})",
        color="grey", linestyle="--"
    )
    # cook's distance contours
    def graph(formula, x_range, label=None):
        x = x_range
        y = formula(x)
        ax.plot(x, y, label=label, lw=1, ls='--', color='red')

    graph(lambda leverage: np.sqrt((cooks_threshold * df_model * (1 - leverage)) / leverage), 
          np.linspace(leverage.min()*0.9, leverage.max(), 1000), 
          f'Cook\'s distance threshold ({cooks_threshold:.2f})')
    graph(lambda leverage: -np.sqrt((cooks_threshold * df_model * (1 - leverage)) / leverage), 
          np.linspace(leverage.min()*0.9, leverage.max(), 1000))
    
    ax.legend(loc='upper right');
    add_margins(ax)


def scale_location(model, ax):

    # normalized residuals
    std_residuals = model.get_influence().resid_studentized_internal
    # absolute squared normalized residuals
    sq_abs_std_residuals = np.sqrt(np.abs(std_residuals))
    data = pd.DataFrame({"y_fitted": model.fittedvalues, "sq_abs_std_residuals": sq_abs_std_residuals})

    sns.regplot(
        data=data,
        x="y_fitted", y="sq_abs_std_residuals", 
        scatter=True, 
        ci=False, 
        lowess=True,
        line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
        scatter_kws=dict(alpha=0.5),
        ax=ax
    )

    ax.set_title('Scale-Location')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('$\sqrt{|Standardized Residuals|}$');

    # annotations    
    data = data.sort_values("sq_abs_std_residuals", ascending=False)
    for i, row in data[:3].iterrows():
        ax.annotate(i, xy=(row.y_fitted, row.sq_abs_std_residuals));
            
def variance_inflation_factors(model):
    exog_df = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
    leave_one_out_R2 = [
        sm.OLS(exog_df[col].values, exog_df.loc[:, exog_df.columns != col].values).fit().rsquared 
        for col in exog_df
        if col != "Intercept"
    ]
    vifs = pd.Series([1 / (1. - R2) for R2 in leave_one_out_R2], index=exog_df.columns[1:], name='VIF')
    return vifs

def rss_contour(model, x_lim, y_lim, ax, levels=None):
    
    def rss(model, beta_x1, beta_x2):
        x1 = model.model.exog[:, 1]
        x2 = model.model.exog[:, 2]
        y = model.model.endog
        return ((y[:, np.newaxis, np.newaxis] - model.params[0] 
                 - x1[:, np.newaxis, np.newaxis]*beta_x1 - x2[:, np.newaxis, np.newaxis]*beta_x2
                )**2).sum(axis=0)/1e6

    N = 100
    x = np.linspace(*x_lim, N)
    y = np.linspace(*y_lim, N)

    X, Y = np.meshgrid(x, y)
    Z = rss(model, X, Y)
    
    beta_x1 = model.params[1]
    beta_x2 = model.params[2]
    
    ax.scatter(x=beta_x1, y=beta_x2, color="black")
    ax.hlines(y=beta_x2, xmin=x_lim[0], xmax=beta_x1, colors="black", linestyles="dashed")
    ax.vlines(x=beta_x1, ymin=y_lim[0], ymax=beta_x2, colors="black", linestyles="dashed")
    
    cplot = ax.contour(X, Y, Z, levels=levels, colors='blue');
    ax.clabel(cplot, inline=True, fontsize=10)
    
    ax.set_xlabel(rf"$\beta_{{{model.model.exog_names[1]}}}$")
    ax.set_ylabel(rf"$\beta_{{{model.model.exog_names[2]}}}$")


def compare_linear_knn(data_train, data_test, data_dense, ax):
    model_linear = smf.ols("y ~ x", data=data_train).fit()
    data_dense["y_linear"] = model_linear.predict(data_dense.x)
    
    data_dense["y_knn_1"] = (
        KNeighborsRegressor(n_neighbors=1).fit(data_train.x.values.reshape(-1, 1), data_train.y)
        .predict(data_dense.x.values.reshape(-1, 1))
    )
    data_dense["y_knn_9"] = (
        KNeighborsRegressor(n_neighbors=9).fit(data_train.x.values.reshape(-1, 1), data_train.y)
        .predict(data_dense.x.values.reshape(-1, 1))
    )
    
    sns.lineplot(data=data_dense, x="x", y="y_linear", color="black", linestyle="--", ax=ax);
    sns.lineplot(data=data_dense, x="x", y="y", color="black", ax=ax)
    sns.lineplot(data=data_dense, x="x", y="y_knn_1", color="blue", lw=1, ax=ax);
    sns.lineplot(data=data_dense, x="x", y="y_knn_9", color="red", lw=1, ax=ax);

    
def linear_knn_mse(X_train, y_train, X_test, y_test, ax):
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)
    model_linear = sm.OLS(y_train, X_train).fit()
    linear_mse = ((model_linear.predict(X_test) - y_test)**2).sum()/len(y_test)
    
    Ks = np.linspace(1, 9, 9)
    knn_mse = []
    for K in Ks:
        knn = KNeighborsRegressor(n_neighbors=int(K)).fit(X_train, y_train)
        mse = ((knn.predict(X_test) - y_test)**2).sum()/len(y_test)
        knn_mse.append(mse)

    data_knn_mse = pd.DataFrame({"K": Ks, "MSE": knn_mse})
    data_knn_mse["inv_K"] = 1/data_knn_mse["K"]
    
    ax.axhline(y=model_linear.mse_resid, color="black", linestyle="--")
    sns.lineplot(data=data_knn_mse, x="inv_K", y="MSE", color="green", markers=True, linestyle="--", ax=ax);
    sns.scatterplot(data=data_knn_mse, x="inv_K", y="MSE", color="green", ax=ax);
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_ylim(bottom=0)
    ax.set_xlabel("1/K");

    
def linear_knn_mse_one_var(data_train, data_test, ax):
    linear_knn_mse(data_train.x.values, data_train.y, data_test.x.values, data_test.y, ax)
