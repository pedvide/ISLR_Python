from sklearn import neighbors
import sklearn.linear_model as skl_lm
from sklearn.utils.validation import check_array
from sklearn.neighbors.base import _get_weights, _check_weights
import numpy as np

class LocalRegressor(neighbors.KNeighborsRegressor):
    
    def predict(self, X):
        """Predict the target for the provided data
        It fits a weighted least squares linear model locally for the nearest neighbors
        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.
        Returns
        -------
        y : array of int, shape = [n_samples] or [n_samples, n_outputs]
            Target values
        """
        X = check_array(X, accept_sparse='csr')

        # NN of X with respect to the train data (_fit_X)
        neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)
        
        _y = self._y
        _fit_X = self._fit_X
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        y_pred = np.empty((X.shape[0], 1), dtype=np.float64)

        # NN in the train data
        X_nn = np.squeeze(_fit_X[neigh_ind])
        y_nn = np.squeeze(_y[neigh_ind])
        
        # loop over the samples, not ideal from a speed point of view
        for i in range(X_nn.shape[0]):
            linear = skl_lm.LinearRegression()
            if weights is not None:
                linear.fit(X_nn[i].reshape(-1, 1), y_nn[i], sample_weight=weights[i])
            else:
                linear.fit(X_nn[i].reshape(-1, 1), y_nn[i])
            y_pred[i] = linear.predict(X[i].reshape(-1, 1))

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred
    