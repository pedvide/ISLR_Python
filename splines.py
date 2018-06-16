# get spline basis and fit it to data using scipy
# from https://gist.github.com/MMesch/35d7833a3daa4a9e8ca9c6953cbe21d4

import numpy as np
import scipy.interpolate as si
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator, RegressorMixin
from patsy import dmatrix


class BSplineFeatures(TransformerMixin):
    '''Cubic splines using scipy'''
    def __init__(self, knots, degree=3, periodic=False):
        self.bsplines = self.get_bspline_basis(knots, degree, periodic=periodic)
        self.nsplines = len(self.bsplines)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nsamples, nfeatures = X.shape
        features = np.zeros((nsamples, nfeatures * self.nsplines))
        for ispline, spline in enumerate(self.bsplines):
            istart = ispline * nfeatures
            iend = (ispline + 1) * nfeatures
            features[:, istart:iend] = si.splev(X, spline)
        return features

    def get_bspline_basis(self, knots, degree=3, periodic=False):
        """Get spline coefficients for each basis spline."""
        nknots = len(knots)
        y_dummy = np.zeros(nknots)

        knots, coeffs, degree = si.splrep(knots, y_dummy, k=degree, s=0,
                                          per=periodic)
        ncoeffs = len(coeffs)
        bsplines = []
        for ispline in range(nknots):
            coeffs = [1.0 if ispl == ispline else 0.0 for ispl in range(ncoeffs)]
            bsplines.append((knots, coeffs, degree))
        return bsplines


class NaturalSplineFeatures(TransformerMixin):
    '''Cubic natural splines using scipy. Not really natural splines yet'''
    def __init__(self, knots, degree=3, periodic=False):
        self.natsplines = self.get_natural_spline_basis(knots, degree, periodic=periodic)
        self.nsplines = len(self.natsplines)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nsamples, nfeatures = X.shape
        features = np.zeros((nsamples, nfeatures * self.nsplines))
        for ispline, spline in enumerate(self.natsplines):
            istart = ispline * nfeatures
            iend = (ispline + 1) * nfeatures
            features[:, istart:iend] = si.splev(X, spline)
        return features

    def get_natural_spline_basis(self, knots, degree=3, periodic=False):
        """Get spline coefficients for each basis spline."""
        nknots = len(knots)
        X_dummy = np.linspace(knots[0], knots[-1], 20*nknots)
        y_dummy = np.zeros_like(X_dummy)

        new_knots, coeffs, degree = si.splrep(X_dummy, y_dummy, k=degree, s=0, t=knots[1:-1], per=periodic)
        ncoeffs = len(coeffs)
        natsplines = []
        for ispline in range(nknots):
            new_coeffs = [1.0 if ispl == ispline else 0.0 for ispl in range(ncoeffs)]
            natsplines.append((new_knots, new_coeffs, degree))
        return natsplines


class PatsySplineFeatures(TransformerMixin):
    '''Cubic splines (natural or bspline) using patsy'''
    def __init__(self, knots=None, df=None, type='natural'):
        '''Either knots or df is required'''
        self.knots = knots
        self.df = df
        if type == 'natural':
            self.function = 'cr'
        elif type == 'bspline':
            self.function = 'bs'
        elif type == 'cyclic':
            self.function = 'cc'
        else:
            raise AttributeError

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        options = 'df=df' if self.df else 'knots=knots'
        features = dmatrix(f"{self.function}(x, {options})", {"x": X, 'knots': self.knots, 
                                                              'df': self.df}, return_type='dataframe')
        return features.values

    
class SmoothingSpline(BaseEstimator, RegressorMixin):
    '''Smoothing cubic splines estimator using scipy'''
    def __init__(self, s=0):
        self.s = s
    
    def fit(self, X, y):
        X_unique, unq_idx, unq_inv, unq_cnt = np.unique(X, return_index=True, return_inverse=True, return_counts=True)
        y_unique_mean = np.bincount(unq_inv, y) / unq_cnt
        self.smoothing_spline_ = si.UnivariateSpline(X_unique, y_unique_mean, s=self.s)
        
        # this should give the effective df of the model
        # but doesn't work
        #pred = smoothing_spline(X_unique).reshape((-1,1))
        #np.trace(np.multiply(pred, y_unique_mean.reshape((-1,1))))
        self.df_ = len(self.smoothing_spline_.get_knots())
        return self
    
    def predict(self, X, y=None):
        return self.smoothing_spline_(X)
    