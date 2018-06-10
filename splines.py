# get spline basis and fit it to data using scipy
# from https://gist.github.com/MMesch/35d7833a3daa4a9e8ca9c6953cbe21d4

import numpy as np
import scipy.interpolate as si
from sklearn.base import TransformerMixin
from patsy import dmatrix


class BSplineFeatures(TransformerMixin):
    '''Works, but it's not a natural spline'''
    def __init__(self, knots, degree=3, periodic=False):
        self.bsplines = get_bspline_basis(knots, degree, periodic=periodic)
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

def get_bspline_basis(knots, degree=3, periodic=False):
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


class CubicSplineFeatures(TransformerMixin):
    '''Doesn't seem to work well'''
    def __init__(self, knots):
        self.cubic_splines = get_cubic_spline_basis(knots)
        self.nsplines = len(self.cubic_splines)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nsamples, nfeatures = X.shape
        features = np.zeros((nsamples, nfeatures * self.nsplines))
        for ispline, spline in enumerate(self.cubic_splines):
            istart = ispline * nfeatures
            iend = (ispline + 1) * nfeatures
            features[:, istart:iend] = si.splev(X, spline)
        return features

def get_cubic_spline_basis(knots):
    """Get spline coefficients for each basis spline."""
    nknots = len(knots)
    y_dummy = np.zeros(nknots)

    cubic_basis = si.CubicSpline(knots, y_dummy, bc_type='natural')
    new_knots = cubic_basis.x
    coeffs = cubic_basis.c
    ncoeffs = len(coeffs)
    cubic_splines = []
    for ispline in range(nknots):
        coeffs = [1.0 if ispl == ispline else 0.0 for ispl in range(ncoeffs)]
        cubic_splines.append((new_knots, coeffs, 3))
    return cubic_splines


class PatsySplineFeatures(TransformerMixin):
    
    def __init__(self, knots, df=None, type='natural'):
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
        features = dmatrix(f"{self.function}(x, {options})", {"x": X, 'knots': self.knots, 'df': self.df}, return_type='dataframe')
        return features.values
