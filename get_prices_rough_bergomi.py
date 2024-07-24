import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# Define utility functions first
def blsimpv(s0, K, y, T, price, q):
    """
    Compute Black-Scholes implied volatility.
    """
    def bs_price(sigma):
        d1 = (np.log(s0 / K) + (y + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return s0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-y * T) * norm.cdf(d2)
    
    implied_vol = np.zeros_like(price)
    for i in range(len(price)):
        if price[i] < blsimpv_bscall(s0, K[i], y[i], T[i], 0.001, q[i]) or price[i] > blsimpv_bscall(s0, K[i], y[i], T[i], 5.0, q[i]):
            implied_vol[i] = np.nan
        else:
            implied_vol[i] = brentq(lambda sigma: bs_price(sigma) - price[i], 0.001, 5.0)
    return implied_vol

def blsimpv_bscall(s0, K, y, T, sigma, q):
    """
    Compute Black-Scholes call price given volatility.
    """
    d1 = (np.log(s0 / K) + (y + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return s0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-y * T) * norm.cdf(d2)

def bscall(S, K, r, T, sigma):
    """
    Compute Black-Scholes call price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bsput(S, K, r, T, sigma):
    """
    Compute Black-Scholes put price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def floor2(value):
    return np.floor(value).astype(int)

# The main function
def get_prices_rough_bergomi(s0, y, q, xi, eta, rho, H, K, T, scheme, N, n, kappa, **kwargs):
    c = kwargs.get('c', None)
    gamm = kwargs.get('gamm', None)
    Z1 = kwargs.get('Z1', None)
    Z2 = kwargs.get('Z2', None)
    turbo = kwargs.get('turbo', True)
    SIGMA = kwargs.get('SIGMA', None)
    conv_method = kwargs.get('conv_method', None)
    explicit = kwargs.get('explicit', False)

    nOpt = len(K)
    uniqT, idxUniqT = np.unique(T, return_index=True)
    maxT = max(uniqT)
    dt = 1 / n

    t_grid_temp = np.arange(0, floor2(n * maxT) + 1) / n
    idxTadj = np.sum(T[:, np.newaxis] >= t_grid_temp - np.finfo(float).eps, axis=1)
    Tadj = t_grid_temp[idxTadj - 1]
    t_grid = t_grid_temp[:max(idxTadj)]
    floor_nmaxT = max(idxTadj) - 1

    if isinstance(xi, CurveClass):
        xi_eval = xi.eval(t_grid[:-1])
    else:
        xi_eval = xi
    if isinstance(y, CurveClass):
        y_eval = y.eval(T)
    else:
        y_eval = y
    if isinstance(q, CurveClass):
        q_eval = q.eval(T)
    else:
        q_eval = q

    if min(T) < 0:
        raise ValueError('GetPricesRoughBergomi: Expiries must be positive.')

    if turbo and N % 2 != 0:
        raise ValueError('GetPricesRoughBergomi: If the "turbo" parameter is set to true the "N" parameter must be divisible by 2.')

    if xi < 0 or eta < 0 or abs(rho) > 1 or H <= 0.0 or H >= 0.5:
        raise ValueError('GetPricesRoughBergomi: One or more model parameters are invalid.')

    if K.ndim > 1 or T.ndim > 1 or K.shape[0] != T.shape[0]:
        raise ValueError('GetPricesRoughBergomi: Input contracts are not specified correctly.')

    F = s0 * np.exp((y_eval - q_eval) * T)
    ZCB = np.exp(-y_eval * T)
    K_adj = K / F
    idxCall = K >= s0

    if turbo:
        N_indep = N // 2
    else:
        N_indep = N

    # Simulate variance process:
    if scheme.lower() == 'hybridtbss':
        Y, _, dW1 = HybridTBSSScheme(N_indep, n, t_grid[-1], np.sqrt(2 * H) * eta, H - 0.5, kappa, Z1, conv_method, [], [], SIGMA)
        if not turbo:
            V = xi_eval * np.exp(Y[:, :-1] - 0.5 * t_grid[:-1] ** (2 * H) * eta ** 2)
        else:
            V = np.zeros((N, floor_nmaxT))
            V[:N // 2, :] = xi_eval * np.exp(Y[:, :-1] - 0.5 * t_grid[:-1] ** (2 * H) * eta ** 2)
            V[N // 2:, :] = xi_eval * np.exp(-Y[:, :-1] - 0.5 * t_grid[:-1] ** (2 * H) * eta ** 2)
    elif scheme.lower() == 'hybridmultifactor':
        if SIGMA is None:
            SIGMA = CovMatrixHybrid(n, kappa, H - 0.5, 'double')
        if kappa > 0:
            w = SIGMA[1:, 0]
        else:
            w = []

        Y, _, dW1 = HybridMultifactorScheme(N_indep, n, t_grid[-1], 0, 0, np.sqrt(2 * H) * eta, gamm, c, kappa, Z=Z1, returndW=True, SIGMA=SIGMA, w=w, explicit=explicit)
        if not turbo:
            V = xi_eval * np.exp(Y.values[:, :-1] - 0.5 * t_grid[:-1] ** (2 * H) * eta ** 2)
        else:
            V = np.zeros((N, floor_nmaxT))
            V[:N // 2, :] = xi_eval * np.exp(Y.values[:, :-1] - 0.5 * t_grid[:-1] ** (2 * H) * eta ** 2)
            V[N // 2:, :] = xi_eval * np.exp(-Y.values[:, :-1] - 0.5 * t_grid[:-1] ** (2 * H) * eta ** 2)
    else:
        raise ValueError('GetPricesRoughBergomi: The chosen scheme is not supported.')

    optPrice = np.zeros(nOpt)
    seOfPrice = np.zeros(nOpt)
    if turbo:
        dlog_S1 = rho * np.sqrt(V) * np.vstack((dW1, -dW1)) - 0.5 * (rho ** 2) * V * dt
        S1 = np.zeros((N, t_grid.size))
        S1[:, 0] = 1
        S1[:, 1:] = np.exp(np.cumsum(dlog_S1, axis=1))

        QV = np.zeros((N, t_grid.size))
        QV[:, 1:] = np.cumsum(V, axis=1) * dt

        for i in range(len(uniqT)):
            idxT = T == uniqT[i]
            idxt = t_grid == Tadj[idxUniqT[i]]
            Ksub = K_adj[idxT]
            idxCallSub = idxCall[idxT]
            nOptSub = len(Ksub)

            totVar_X = (1 - rho ** 2) * QV[:, idxt]
            X_cond_mc = np.zeros((N, nOptSub))
            X_cond_mc[:, idxCallSub] = bscall(S1[:, idxt], Ksub[idxCallSub], 0, 1, totVar_X)
            X_cond_mc[:, ~idxCallSub] = bsput(S1[:, idxt], Ksub[~idxCallSub], 0, 1, totVar_X)

            X = 0.5 * (X_cond_mc[:N // 2, :] + X_cond_mc[N // 2:, :])
            Y_cv = 0.5 * (S1[:N // 2, idxt] + S1[N // 2:, idxt])

            mu_X = np.mean(X, axis=0)
            mu_Y = np.mean(Y_cv, axis=0)
            diff_Y = Y_cv - mu_Y
            alpha = -np.sum((X - mu_X) * diff_Y, axis=0) / np.sum(diff_Y ** 2, axis=0)
            alpha[np.isnan(alpha)] = 0

            Z = ZCB[idxT] * F[idxT] * (X + (Y_cv - 1) * alpha)
            optPrice[idxT] = np.mean(Z, axis=0)
            seOfPrice[idxT] = np.std(Z, axis=0) / np.sqrt(Z.shape[0])
    else:
        if Z1 is None:
            dWperp = np.sqrt(dt) * np.random.randn(*dW1.shape)
        else:
            if Z2.shape != (N, floor_nmaxT):
                raise ValueError('GetPricesRoughBergomi: The dimensions of the "Z2" parameter are invalid.')
            dWperp = np.sqrt(dt) * Z2
        dW2 = rho * dW1 + np.sqrt(1 - rho ** 2) * dWperp
        dlog_S = np.sqrt(V) * dW2 - 0.5 * V * dt
        S = np.zeros((N, t_grid.size))
        S[:, 0] = 1
        S[:, 1:] = np.exp(np.cumsum(dlog_S, axis=1))

        for i in range(len(uniqT)):
            idxT = T == uniqT[i]
            idxt = t_grid == Tadj[idxUniqT[i]]
            Ksub = K_adj[idxT]
            Fsub = F[idxT]
            ZCBsub = ZCB[idxT]
            idxCallSub = idxCall[idxT]

            Z = np.zeros((np.sum(idxT), N))
            Z[idxCallSub, :] = ZCBsub[idxCallSub] * Fsub[idxCallSub] * np.maximum(S[:, idxt].T - Ksub[idxCallSub], 0)
            Z[~idxCallSub, :] = ZCBsub[~idxCallSub] * Fsub[~idxCallSub] * np.maximum(Ksub[~idxCallSub] - S[:, idxt].T, 0)
            optPrice[idxT] = np.mean(Z, axis=1)
            seOfPrice[idxT] = np.std(Z, axis=1) / np.sqrt(Z.shape[1])

    iv = blsimpv(s0, K, y_eval, T, optPrice, q_eval)
    se = seOfPrice / bsgreek_vega(s0, K, y_eval, T, iv, q_eval)

    return iv, se

