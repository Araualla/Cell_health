import numpy as np
import pandas as pd

pca_matrix_fcn={
    'cov':np.ma.cov,
    'corrcoef':np.ma.corrcoef,
}
class PartialPCA():
    """Finds PCA components for partially observed data"""

    def __init__(self, pca_type='cov', n_components=None):
        self.n_components = n_components
        self.pca_type = pca_type

    def fit(self, X):
        # calculate mean
        mean_vector = np.nanmean(X, axis=0)
        # center X
        centered_X = X - mean_vector
        # mask nans
        centered_X = np.ma.array(centered_X, mask=np.isnan(centered_X))
        # calc covariance matrix
        cov_matrix = np.array(pca_matrix_fcn[self.pca_type](centered_X.T))
        self.cov_matrix = cov_matrix
        # eigenvectors and eigenvalues for the from the scatter matrix
        eig_val, eig_vec = np.linalg.eig(cov_matrix)
        # sort by decreasing eigenvalue
        # Make a list of (eigenvalue, eigenvector) tuples
        self.eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(len(eig_val))]
        # Sort the (eigenvalue, eigenvector) tuples from high to low
        self.eig_pairs.sort(key=lambda x: x[0], reverse=True)
        self.eig_val = [x[0] for x in self.eig_pairs]
        self.eig_vec = [x[1] for x in self.eig_pairs]
        self.loadings = [np.sqrt(val)*vec for val,vec in self.eig_pairs]
        if self.n_components is None: self.n_components = X.shape[1]
        # pick the first n_components eigenvectors
        self.w = np.hstack([self.eig_pairs[j][1].reshape(X.shape[1], 1) for j in range(self.n_components)])
        self.X = X

    """change the number of components used for the PCA
    (can be used only after self.fit() has been run once)"""

    def change_components(self, n_components):
        self.n_components = n_components
        self.w = np.hstack([self.eig_pairs[j][1].reshape(X.shape[1], 1) for j in range(self.n_components)])

    """transform the matrix interpolating the nan values first"""

    def transform(self, X, interp_method='linear', fill_method='mean'):
        # do linear interpolation (fast with pandas)
        if interp_method is not None:
            X_fill = pd.DataFrame(X).interpolate(method=interp_method)
        else:
            x_fill = pd.Dataframe(X)
        # if data missing from start or end of timeseries,
        # linear interpolation will return nans. Pad these
        # nan values with something.
        if fill_method is not None:
            if fill_method == 'mean':
                X_fill = X_fill.fillna(X_fill.mean())
            else:
                X_fill = X_fill.fillna(method=fill_method)
        self.transformed_X = self.w.T.dot(X_fill.values.T).T
        return self.transformed_X

    def explained_variance(self):
        covar = np.cov(self.transformed_X.T)
        variances = np.diag(covar)
        return variances / np.sum(variances)

    """inverse transform of w matrix"""

    def inv_transform(self, X_transformed):
        # check that it;s a square matrix
        # inverse will work only if n_components is equal to n of columns in original data
        assert self.w.shape[0] == self.w.shape[1]
        self.w_inv = np.linalg.inv(self.w)  # otherwise matrix not square!
        return X_transformed.dot(self.w_inv)
