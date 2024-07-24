import numpy as np
from scipy.linalg import cholesky
from numpy import floor

def floor2(value):
    return np.floor(value).astype(int)

def locate_timepoints_in_discretisation(t_grid, t_points):
    return np.searchsorted(t_grid, t_points, side='right') - 1

def chol_mod(Sigma, tol):
    """Modified Cholesky decomposition with a tolerance for positive semi-definite matrices."""
    try:
        return cholesky(Sigma, lower=True)
    except np.linalg.LinAlgError:
        # Fallback for positive semi-definite matrices
        n = Sigma.shape[0]
        L = np.zeros_like(Sigma)
        for i in range(n):
            L[i, i] = max(Sigma[i, i] - tol, 0)
            for j in range(i+1, n):
                L[j, i] = Sigma[j, i]
        return L

def convert_matrix(mat, precision):
    if precision == 'single':
        return mat.astype(np.float32)
    else:
        return mat.astype(np.float64)



def hybrid_multifactor_scheme(N, n, T, g, b, sigma, gamm, c, kappa, **kwargs):
    SIGMA = kwargs.get('SIGMA', None)
    w = kwargs.get('w', None)
    K = kwargs.get('K', None)
    returnU = kwargs.get('returnU', False)
    returndW = kwargs.get('returndW', False)
    tX = kwargs.get('tX', None)
    tU = kwargs.get('tU', None)
    precision = kwargs.get('precision', 'double')
    Z = kwargs.get('Z', None)
    W_bold = kwargs.get('W_bold', None)
    positive = kwargs.get('positive', False)
    explicit = kwargs.get('explicit', False)
    
    if not T and not tX:
        raise ValueError("You must use either the 'T' parameter or the 'tX' parameter.")
    if tX is not None and T is not None:
        raise ValueError("You cannot use both the 'T' and 'tX' parameter at the same time.")
    if tU is not None and T is not None:
        raise ValueError("You cannot use both the 'T' and 'tU' parameter at the same time.")
    if tU is not None and not returnU:
        raise ValueError("If the 'tU' parameter is used, the 'returnU' parameter must be set to true.")
    if returnU and not T and not tU:
        raise ValueError("If returnU is true, then either the 'T' parameter or the 'tU' parameter must be used.")
    
    if tX is not None:
        if any(np.diff(tX) <= 0):
            raise ValueError("The values of the 'tX' vector must be strictly increasing.")
        if tX[0] < 0:
            raise ValueError("Negative time points are not allowed.")
    if tU is not None:
        if any(np.diff(tU) <= 0):
            raise ValueError("The values of the 'tU' vector must be strictly increasing.")
        if tU[0] < 0:
            raise ValueError("Negative time points are not allowed.")
    
    if not T:
        T = max(np.concatenate((tX, tU)))
    
    m = gamm.shape[0]
    dt = 1 / n
    gamm = gamm.T
    c = c.T
    
    floor_nT = floor2(n * T)
    
    if floor_nT <= 0:
        raise ValueError("Parameters must be such that floor(n*T) > 0.")
    
    M = floor_nT + 1
    t_grid = np.arange(0, M) / n
    
    if tX is None:
        tX = t_grid
    if tU is None and returnU:
        tU = t_grid
    elif tU is None and not returnU:
        tU = []
    
    MX = tX.shape[0]
    MU = tU.shape[0]
    
    idxX = locate_timepoints_in_discretisation(t_grid, tX)
    idxU = locate_timepoints_in_discretisation(t_grid, tU) if tU.size > 0 else []
    tX_trunc = t_grid[idxX]
    tU_trunc = t_grid[idxU]
    
    Xpaths = {'t': tX, 't_truncated': tX_trunc, 'values': np.zeros((N, MX), dtype=precision)}
    Upaths = {'t': tU, 't_truncated': tU_trunc, 'values': np.zeros((N, m, MU), dtype=precision) if returnU else None}
    
    b_is_mat = isinstance(b, np.ndarray) and b.shape == (N, M-1)
    b_is_function = callable(b)
    b_is_constant = np.isscalar(b)
    
    sigma_is_mat = isinstance(sigma, np.ndarray) and sigma.shape == (N, M-1)
    sigma_is_function = callable(sigma)
    sigma_is_constant = np.isscalar(sigma)
    
    if callable(g):
        g = g(t_grid)
    elif np.isscalar(g):
        g = np.full(M, g)
    
    f_adj = np.maximum if positive else lambda x: x
    
    if Z is not None and W_bold is not None:
        raise ValueError("You cannot specify both the 'Z' and 'W_bold' parameter at the same time.")
    
    if not np.issubdtype(type(kappa), np.integer) or kappa < 0 or kappa > floor_nT:
        raise ValueError("Kappa must be an integer between 0 and floor(n*T).")
    
    X = np.full(N, g[0], dtype=precision)
    
    U = np.zeros((N, m), dtype=precision)
    
    if W_bold is None:
        if SIGMA is None:
            if K is None and kappa > 0:
                raise ValueError("When the covariance matrix SIGMA is not inputted, the W_bold parameter is unused, and kappa > 0, the kernel function K must be inputted instead.")
            SIGMA = convert_matrix(get_volterra_covariance_matrix(K, kappa, dt), precision)
        else:
            SIGMA = convert_matrix(SIGMA, precision)
        
        A = chol_mod(SIGMA, 1e-14)
        
        if Z is None:
            Z = np.random.randn((M-1)*N, kappa+1, dtype=precision)
        else:
            Z = convert_matrix(Z, precision)
        
        W_bold = Z @ A.T
        W_bold = W_bold.reshape(N, M-1, kappa+1)
    else:
        W_bold = convert_matrix(W_bold, precision)
    
    if b_is_function:
        b_mat = np.zeros((N, M-1), dtype=precision)
    elif b_is_mat:
        b_mat = convert_matrix(b, precision)
    
    if sigma_is_function:
        sigma_mat = np.zeros((N, M-1), dtype=precision)
    elif sigma_is_mat:
        sigma_mat = convert_matrix(sigma, precision)
    
    if kappa > 0:
        if w is None:
            if K is None:
                raise ValueError("When the w-vector is not inputted (and kappa > 0) the kernel function K must be inputted instead.")
            w = convert_matrix(K.integrate(np.arange(0, kappa) * dt, np.arange(1, kappa+1) * dt, 1, dt, c, gamm), precision).T
        else:
            w = convert_matrix(w.T, precision)
    
    dummy1 = convert_matrix(c * np.exp(-gamm * kappa * dt), precision)
    dummy2 = convert_matrix(1 / (1 + gamm * dt), precision)
    
    X_stored_in_Xpaths = False
    for i in range(M-1+returnU*kappa):
        if i <= M-1:
            if b_is_function:
                if X_stored_in_Xpaths:
                    b_mat[:, i] = b(t_grid[i], f_adj(Xpaths['values'][:, idxXi]))
                else:
                    b_mat[:, i] = b(t_grid[i], f_adj(X))
            if sigma_is_function:
                if X_stored_in_Xpaths:
                    sigma_mat[:, i] = sigma(t_grid[i], f_adj(Xpaths['values'][:, idxXi]))
                else:
                    sigma_mat[:, i] = sigma(t_grid[i], f_adj(X))
        
        if i > kappa:
            if not b_is_constant and not sigma_is_constant:
                if explicit:
                    U = U * (1 - gamm * dt) + b_mat[:, i-kappa] * dt + sigma_mat[:, i-kappa] * W_bold[:, i-kappa, 0]
                else:
                    U = dummy2 * (U + b_mat[:, i-kappa] * dt + sigma_mat[:, i-kappa] * W_bold[:, i-kappa, 0])
            elif b_is_constant and not sigma_is_constant:
                if explicit:
                    U = U * (1 - gamm * dt) + b * dt + sigma_mat[:, i-kappa] * W_bold[:, i-kappa, 0]
                else:
                    U = dummy2 * (U + b * dt + sigma_mat[:, i-kappa] * W_bold[:, i-kappa, 0])
            elif not b_is_constant and sigma_is_constant:
                if explicit:
                    U = U * (1 - gamm * dt) + b_mat[:, i-kappa] * dt + sigma * W_bold[:, i-kappa, 0]
                else:
                    U = dummy2 * (U + b_mat[:, i-kappa] * dt + sigma * W_bold[:, i-kappa, 0])
            elif b_is_constant and sigma_is_constant:
                if explicit:
                    U = U * (1 - gamm * dt) + b * dt + sigma * W_bold[:, i-kappa, 0]
                else:
                    U = dummy2 * (U + b * dt + sigma * W_bold[:, i-kappa, 0])
            
            if returnU:
                idxUi = np.isin(idxU, i+1-kappa)
                if any(idxUi):
                    Upaths['values'][:, :, idxUi] = np.tile(U, (1, 1, np.sum(idxUi)))
        
        if i <= M-1:
            idxXi = np.isin(idxX, i+1)
            if not any(idxXi) and not b_is_function and not sigma_is_function:
                continue
            elif not any(idxXi) and (b_is_function or sigma_is_function):
                X_stored_in_Xpaths = False
            elif any(idxXi):
                X_stored_in_Xpaths = True
            
            if kappa == 0:
                if X_stored_in_Xpaths:
                    Xpaths['values'][:, idxXi] = np.tile(f_adj(g[:, i+1] + np.sum(c * U, axis=1)), (1, 1, np.sum(idxXi)))
                else:
                    X = f_adj(g[:, i+1] + np.sum(c * U, axis=1))
            else:
                sigma_term = 0
                for k in range(1, min(kappa, i)+1):
                    if not sigma_is_constant:
                        sigma_term += sigma_mat[:, i-k+1] * W_bold[:, i-k+1, 1+k]
                    else:
                        sigma_term += sigma * W_bold[:, i-k+1, 1+k]
                kIdx = np.arange(1, min(kappa, i)+1)
                if i <= kappa:
                    if not b_is_constant:
                        if X_stored_in_Xpaths:
                            Xpaths['values'][:, idxXi] = np.tile(f_adj(g[:, i+1] + np.sum(b_mat[:, i-kIdx+1] * w[kIdx-1], axis=1) + sigma_term), (1, 1, np.sum(idxXi)))
                        else:
                            X = f_adj(g[:, i+1] + np.sum(b_mat[:, i-kIdx+1] * w[kIdx-1], axis=1) + sigma_term)
                    else:
                        if X_stored_in_Xpaths:
                            Xpaths['values'][:, idxXi] = np.tile(f_adj(g[:, i+1] + b * np.sum(w[kIdx-1], axis=1) + sigma_term), (1, 1, np.sum(idxXi)))
                        else:
                            X = f_adj(g[:, i+1] + b * np.sum(w[kIdx-1], axis=1) + sigma_term)
                else:
                    weighted_Us = dummy1 * U
                    if not b_is_constant:
                        if X_stored_in_Xpaths:
                            Xpaths['values'][:, idxXi] = np.tile(f_adj(g[:, i+1] + np.sum(weighted_Us, axis=1) + np.sum(b_mat[:, i-kIdx+1] * w[kIdx-1], axis=1) + sigma_term), (1, 1, np.sum(idxXi)))
                        else:
                            X = f_adj(g[:, i+1] + np.sum(weighted_Us, axis=1) + np.sum(b_mat[:, i-kIdx+1] * w[kIdx-1], axis=1) + sigma_term)
                    else:
                        if X_stored_in_Xpaths:
                            Xpaths['values'][:, idxXi] = np.tile(f_adj(g[:, i+1] + np.sum(weighted_Us, axis=1) + b * np.sum(w[kIdx-1], axis=1) + sigma_term), (1, 1, np.sum(idxXi)))
                        else:
                            X = f_adj(g[:, i+1] + np.sum(weighted_Us, axis=1) + b * np.sum(w[kIdx-1], axis=1) + sigma_term)
    
    dW = np.squeeze(W_bold[:, :, 0]) if returndW else None
    
    return Xpaths, Upaths, dW

