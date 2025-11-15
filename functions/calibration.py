import numpy as np
import scipy.optimize as optimize
import time
import optuna
from scipy.optimize import OptimizeResult
optuna.logging.set_verbosity(optuna.logging.ERROR)
from tqdm import tqdm

def find_initial_values(smile, beta):
    """
    Calculates the initial parameters for SABR smile calibration following the methods of Le Floc'h and Kennedy (2014)
    Based on Stuijt (2021): https://github.com/hugostuijt/deep-SABR-volatility-calibration/blob/main/Code/src/SABR.py
    """
    default_output  = [.05, -.5, .5]
    fw = smile['forward_price'][0]

    offset = .1
    atm = smile.iloc[(smile['strike_price'] - fw).abs().argsort()[0]]
    below = smile[smile['strike_price'] < atm['strike_price']]
    if len(below) == 0:
        print('No strikes under ATM')
        return default_output
    below = below.iloc[(below['strike_price'] - (1 - offset) * fw).abs().argsort().reset_index(drop=True)[0]]

    up = smile[smile['strike_price'] > atm['strike_price']]
    if len(up) == 0:
        print('No strikes after ATM')
        return default_output
    up = up.iloc[(up['strike_price'] - (1 + offset) * fw).abs().argsort().reset_index(drop=True)[0]]
    z_min = np.log(below['strike_price'] / fw)  # z[0]
    z_0 = np.log(atm['strike_price'] / fw)  # z[1]
    z_plus = np.log(up['strike_price'] / fw)  # z[2]

    sigma_min = below['impl_volatility']  # sigma[0]
    sigma_0 = atm['impl_volatility']  # sigma[1]
    sigma_plus = up['impl_volatility']  # sigma[2]

    w_min = 1 / ((z_min - z_0) * (z_min - z_plus))
    w_0 = 1 / ((z_0 - z_min) * (z_0 - z_plus))
    w_plus = 1 / ((z_plus - z_min) * (z_plus - z_0))

    s = z_0 * z_plus * w_min * sigma_min + z_min * z_plus * w_0 * sigma_0 + z_min * z_0 * w_plus * sigma_plus
    s_ = -(z_0 + z_plus) * w_min * sigma_min - (z_min + z_plus) * w_0 * sigma_0 - (z_min + z_0) * w_plus * sigma_plus
    s__ = 2 * w_min * sigma_min + 2 * w_0 * sigma_0 + 2 * w_plus * sigma_plus

    alpha = s * fw ** (1 - beta)
    v_sq = 3 * s * s__ - 0.5 * (1 - beta) ** 2 * s ** 2 + 1.5 * (2 * s_ + (1 - beta) * s) ** 2

    if v_sq < 0:
        rho = min(.98 * np.sign(2 * s_ + (1 - beta) * s), 0.45)
        v = (2 * s_ + (1 - beta) * s) / rho

        output = [alpha, rho, v]
        if np.nan in output or np.inf in np.abs(output):
            return default_output
        return output

    v = np.sqrt(v_sq)
    rho = min(((2 * s_ + (1 - beta) * s) / v), 0.45)
    if np.abs(rho) > 1:
        rho = .98 * np.sign(rho)
    output = [alpha, rho, v]

    if np.nan in output or np.inf in np.abs(output):
        return default_output
    return output

def residuals(SABR_params, chunk, approximator, T, beta):
    """
    SABR residuals used for solving calibration. 
    Based on Stuijt (2021): https://github.com/hugostuijt/deep-SABR-volatility-calibration/blob/main/Code/src/SABR.py
    """
    alpha, rho, v = SABR_params
    # create artificial bounds
    if alpha < 0 or v < 0 or abs(rho) > 1:
        return np.array([np.inf] * len(chunk))
    approximation = chunk.apply(
        lambda x: approximator.calc_iv(alpha, beta, rho, v, (x['strike_price'], x['forward_price']), T), axis=1)
    error = chunk['impl_volatility'] - approximation
    return error

def calibrate_smile(chunk, approximator, beta=0.5, n_trials=300, loss_threshold=0.0):
    """
    Volatility curve calibration function.
    """

    T = chunk['T'][0]

    alpha0, rho0, v0 = find_initial_values(chunk, beta)
    
    def objective(trial):
        try:
            alpha = trial.suggest_float('alpha', 0.005, 1.0, log=True)
            rho = trial.suggest_float('rho', -1.0, 0.5)
            v     = trial.suggest_float('v', 0.0001, 6.0, log=True)

            res = residuals([alpha, rho, v], chunk, approximator, T, beta)
            if res.isna().any():
                loss = np.nan
            else:
                loss = np.mean(res ** 2)

            if np.isnan(loss) or np.isinf(loss):
                raise ValueError("Invalid loss value")

            # Early stopping condition
            if loss < loss_threshold:
                study.stop()

            return loss

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return float('inf')

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())

    study.enqueue_trial({
    'alpha': float(alpha0),
    'rho': float(rho0),
    'v': float(v0)
    })

    start = time.time()
    study.optimize(objective, n_trials=n_trials)
    stop = time.time() - start

    best = study.best_params
    result = OptimizeResult(
        x=np.array([best['alpha'], best['rho'], best['v']]),
        success=True,
        fun=study.best_value
    )

    return result, stop

def apply_approximator(chunks, approximator_class, approximator_name):
    """
    Applies a SABR approximator to a list of dataframes, each representing one implied volatility curve.
    Computes the corresponding implied volatilities from the provided SABR parameters.
    """
    for i, chunk in tqdm(enumerate(chunks), desc="Simulating"):
        alpha = float(chunk['alpha'].iloc[0])
        rho = float(chunk['rho'].iloc[0])
        v = float(chunk['nu'].iloc[0])
        beta = 0.5
        F0 = float(chunk['forward_price'].iloc[0])
        T = float(chunk['T'].iloc[0])

        approximator = approximator_class()

        chunk[f"impl_volatility_{approximator_name}"] = [
            approximator.calc_iv(alpha, beta, rho, v, (K, F0), T)
            for K in chunk['strike_price']
        ]


def split_dataframe(df, chunk_size):
    """
    Splits dataframe into list of dataframes (chunks)
    """
    return [df.iloc[i:i + chunk_size].copy().reset_index(drop=True) for i in range(0, len(df), chunk_size)]

def process_chunk(i, chunk, model, approximator_name):
    """
    Helperfunction for calibrating implied volatility curves in dataframe format and adding calibrated parameters and implied volatilities to respective dataframe.
    """
    params = calibrate_smile(chunk, model, beta=0.5)
    params_dict, _ = params
    x_params = params_dict['x']
    param_names = ['alpha', 'rho', 'v']
    param_dict = {name: value for name, value in zip(param_names, x_params)}
    chunk[f'params_{approximator_name}'] = [param_dict] * len(chunk)

    alpha, rho, v = param_dict['alpha'], param_dict['rho'], param_dict['v']
    F0 = chunk['forward_price'].iloc[0]
    T = chunk['T'].iloc[0]

    chunk[f'impl_volatility_{approximator_name}'] = [
        model.calc_iv(alpha, 0.5, rho, v, (K, F0), T)
        for K in chunk['strike_price']
    ]
    return chunk

def IV_derivative(chunks, approximator_class, approximator_name):
    """
    Computes dervivatives with reagrd to F, alpha, nu, and rho for each dataframe in a list using central differences.
    Then stores the Greeks in the same dataframe.
    """
    processed_chunks = []
    skipped_chunks = []

    for i, chunk in tqdm(enumerate(chunks), desc="Simulating"):
        try:
            alpha = float(chunk[f'alpha_{approximator_name}'].iloc[0])
            rho   = float(chunk[f'rho_{approximator_name}'].iloc[0])
            v     = float(chunk[f'nu_{approximator_name}'].iloc[0])
            beta  = 0.5
            F0     = float(chunk['forward_price'].iloc[0])
            T     = float(chunk['T'].iloc[0])

            approximator = approximator_class()

            # Compute implied vols
            chunk[f"impl_volatility_{approximator_name}"] = [
                approximator.calc_iv(alpha, beta, rho, v, (K, F0), T)
                for K in chunk['strike_price']
            ]

            # Perturbations
            dF   = F0 * 0.0025
            dA   = alpha * 0.0025
            dNu  = v * 0.0025
            dRho = rho * 0.0025

            # Central differences for F
            iv_plus_F = [
                approximator.calc_iv(alpha, beta, rho, v, (K, F0 + dF), T)
                for K in chunk['strike_price']
            ]
            iv_minus_F = [
                approximator.calc_iv(alpha, beta, rho, v, (K, F0 - dF), T)
                for K in chunk['strike_price']
            ]
            chunk[f"dIV_dF_{approximator_name}"] = [
                (ivp - ivm) / (2 * dF) for ivp, ivm in zip(iv_plus_F, iv_minus_F)
            ]

            # Central differences for alpha
            iv_plus_A = [
                approximator.calc_iv(alpha + dA, beta, rho, v, (K, F0), T)
                for K in chunk['strike_price']
            ]
            iv_minus_A = [
                approximator.calc_iv(alpha - dA, beta, rho, v, (K, F0), T)
                for K in chunk['strike_price']
            ]
            chunk[f"dIV_dalpha_{approximator_name}"] = [
                (ivp - ivm) / (2 * dA) for ivp, ivm in zip(iv_plus_A, iv_minus_A)
            ]

            # Central differences for nu
            iv_plus_nu = [
                approximator.calc_iv(alpha, beta, rho, v + dNu, (K, F0), T)
                for K in chunk['strike_price']
            ]
            iv_minus_nu = [
                approximator.calc_iv(alpha, beta, rho, v - dNu, (K, F0), T)
                for K in chunk['strike_price']
            ]
            chunk[f"dIV_dnu_{approximator_name}"] = [
                (ivp - ivm) / (2 * dNu) for ivp, ivm in zip(iv_plus_nu, iv_minus_nu)
            ]

            # Central differences for rho
            iv_plus_rho = [
                approximator.calc_iv(alpha, beta, rho + dRho, v, (K, F0), T)
                for K in chunk['strike_price']
            ]
            iv_minus_rho = [
                approximator.calc_iv(alpha, beta, rho - dRho, v, (K, F0), T)
                for K in chunk['strike_price']
            ]
            chunk[f"dIV_drho_{approximator_name}"] = [
                (ivp - ivm) / (2 * dRho) for ivp, ivm in zip(iv_plus_rho, iv_minus_rho)
            ]

            processed_chunks.append(chunk)

        except Exception:
            skipped_chunks.append(i)
            continue

    return processed_chunks, skipped_chunks