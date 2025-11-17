import numpy as np
import scipy.integrate as integrate
import time
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.serialization import safe_globals

# Based on Jäckel (2015)
from py_vollib_vectorized import vectorized_implied_volatility
from py_vollib.black import implied_volatility
# import cupy as cp

# The following functions implement the Antonov et al. (2013) approximation.  
# Variable names follow the original notation used in Antonov et al. (2013) for clarity and consistency.
def phi(s, s_min, s_plus):
    """
    Equation (C.11)
    Based on Stuijt (2021): https://github.com/hugostuijt/deep-SABR-volatility-calibration/blob/main/Code/src/SABR_IV_approximators.py
    """
    num = np.sinh(s) ** 2 - np.sinh(s_min) ** 2
    denom = np.sinh(s_plus) ** 2 - np.sinh(s) ** 2
    return 2 * np.arctan(np.sqrt(num / denom))


def psi(s, s_min, s_plus):
    """
    Equation (C.11)
    Based on Stuijt (2021): https://github.com/hugostuijt/deep-SABR-volatility-calibration/blob/main/Code/src/SABR_IV_approximators.py
    """
    num = np.sinh(s) ** 2 - np.sinh(s_plus) ** 2
    denom = np.sinh(s) ** 2 - np.sinh(s_min) ** 2

    if num == np.inf:
        print('Error in psi')
        num = 1
        denom = 1
    output = 2 * np.arctanh(np.sqrt(num / denom))
    return output


def kernel_approx(tau, s):
    """
    Equations (C.14-15)
    Based on Stuijt (2021): https://github.com/hugostuijt/deep-SABR-volatility-calibration/blob/main/Code/src/SABR_IV_approximators.py
    """
    coth_s = np.cosh(s) / np.sinh(s)
    g_s = s * coth_s - 1

    frac1 = 3 * tau * g_s / (8 * s ** 2)
    frac2 = 5 * tau ** 2 * (-8 * s ** 2 + 3 * g_s ** 2 + 24 * g_s) / (128 * s ** 4)
    frac3 = 35 * tau ** 3 * (-40 * s ** 2 + 3 * g_s ** 3 + 24 * g_s ** 2 + 120 * g_s) / (1024 * s ** 6)
    R = 1 + frac1 - frac2 + frac3

    frac = (3072 + 384 * tau + 24 * tau ** 2 + tau ** 3) / 3072
    delta_R = np.exp(tau / 8) - frac

    return np.sqrt(np.sinh(s) / s) * np.exp(-s ** 2 / (2 * tau) - tau / 8) * (R + delta_R)




def integral1_approx(s, eta, tau, s_min, s_plus):
    """
    part of equation (C.8)
    Based on Stuijt (2021): https://github.com/hugostuijt/deep-SABR-volatility-calibration/blob/main/Code/src/SABR_IV_approximators.py
    """
    frac = np.sin(eta * phi(s, s_min, s_plus)) / np.sinh(s)

    return frac * kernel_approx(tau, s)


def integral2_approx(s, eta, tau, s_min, s_plus):
    """
    part of Equation (C.8)
    Based on Stuijt (2021): https://github.com/hugostuijt/deep-SABR-volatility-calibration/blob/main/Code/src/SABR_IV_approximators.py
    """
    frac = np.exp(-eta * psi(s, s_min, s_plus)) / np.sinh(s)

    return frac * kernel_approx(tau, s)


def calc_efficient_params(v0, beta, rho, gamma, contract_params, T):
    """
    Equations (C.1-7)
    Based on Stuijt (2021): https://github.com/hugostuijt/deep-SABR-volatility-calibration/blob/main/Code/src/SABR_IV_approximators.py
    """
    K, F0 = contract_params

    beta_tilde = beta
    gamma_tilde_squared = gamma ** 2 - 3 * ((gamma * rho) ** 2 + v0 * gamma * rho * (1 - beta) * (F0 ** (beta - 1))) / 2
    gamma_tilde = np.sqrt(gamma_tilde_squared)

    delta_q = ((K ** (1 - beta)) - (F0 ** (1 - beta))) / (1 - beta)
    v_min_squared = (gamma * delta_q) ** 2 + 2 * rho * gamma * delta_q * v0 + v0 ** 2
    v_min = np.sqrt(v_min_squared)

    # calculation of Phi
    num = v_min + rho * v0 + gamma * delta_q
    denom = v0 * (1 + rho)
    Phi = (num / denom) ** (gamma_tilde / gamma)

    delta_q_tilde = ((K ** (1 - beta_tilde)) - (F0 ** (1 - beta_tilde))) / (1 - beta_tilde)

    # calculation of v0_tilde
    num = 2 * Phi * delta_q_tilde * gamma_tilde
    denom = Phi ** 2 - 1
    v0_tilde_0 = num / denom

    # calculation of u0
    num = delta_q * gamma * rho + v0 - v_min
    denom = delta_q * gamma * np.sqrt(1 - rho * rho)
    u0 = num / denom

    q = K ** (1 - beta) / (1 - beta)  # edit
    # calculation of L
    num = v_min  # * (1 - beta)
    denom = (K ** (1 - beta)) * gamma * np.sqrt(1 - rho * rho)
    denom = q * gamma * np.sqrt(1 - rho * rho)
    L = num / denom

    # calculation of I
    if L < 1:
        denom = np.sqrt(1 - L * L)
        I_f = (2 / denom) * (np.arctan((u0 + L) / denom) - np.arctan(L / denom))
    elif L > 1:
        denom = np.sqrt(L * L - 1)
        I_f = (1 / denom) * np.log((u0 * (L + denom) + 1) / (u0 * (L - denom) + 1))
    else:
        print(v0)
        print(beta)
        print(rho)
        print(gamma)
        print(contract_params)
        print(T)
        raise Exception('Errors with calculation of L')

    # calculation of phi_0
    num = delta_q * gamma + v0 * rho
    phi0 = np.arccos(-num / v_min)

    B_min = -0.5 * (beta / (1 - beta)) * (rho / np.sqrt(1 - rho * rho)) * (np.pi - phi0 - np.arccos(rho) - I_f)

    # Formulas from Antonov (2015),
    v_min_tilde2 = (gamma_tilde * delta_q) ** 2 + v0_tilde_0 ** 2
    v_min_tilde = np.sqrt(v_min_tilde2)
    R_tilde = delta_q * gamma_tilde / v0_tilde_0
    h = np.sqrt(1 + R_tilde ** 2)
    num = 0.5 * np.log(v0 * v_min / (v0_tilde_0 * v_min_tilde)) - B_min
    denom = R_tilde * np.log(h + R_tilde)
    v0_tilde_1 = v0_tilde_0 * (gamma_tilde ** 2) * h * num / denom

    v0_tilde = v0_tilde_0 + T * v0_tilde_1

    return v0_tilde, beta_tilde, gamma_tilde

class Antonov():
    """
    Antonov et al. (2013) Approximation of the SABR model.
    Equations (C.8), (C.9), (C.10), (C.12) part of (C.3)
    Based on Stuijt (2021): https://github.com/hugostuijt/deep-SABR-volatility-calibration/blob/main/Code/src/SABR_IV_approximators.py
    """

    def get_name(self):
        return 'Antonov Approx'

    def calc_call(self, alpha, beta, rho, v, contract_params, T):
        K, F0 = contract_params

        # Switch to Antonov (2013) notation
        gamma = v
        v0 = alpha

        # Calculate the parameters of the mimicking zero correlation SABR model.
        if rho != 0:
            v0, beta, gamma = calc_efficient_params(v0, beta, rho, gamma, contract_params, T)

        # call option approximation
        q = (K ** (1 - beta)) / (1 - beta)
        q0 = (F0 ** (1 - beta)) / (1 - beta)

        s_min = np.arcsinh((gamma * np.abs(q - q0)) / v0)
        s_plus = np.arcsinh((gamma * (q + q0)) / v0)

        eta = np.abs(1 / (2 * (beta - 1)))
        tau = T * gamma * gamma

        if np.nan in [s_min, s_plus, eta, v]:
            return np.inf

        # compute the integrals
        integral1_eval = integrate.quad(lambda s: integral1_approx(s, eta, tau, s_min, s_plus), a=s_min, b=s_plus)
        integral2_eval = integrate.quad(lambda s: integral2_approx(s, eta, tau, s_min, s_plus), a=s_plus, b=200)

        call_price = max(F0 - K, 0) + (2 / np.pi) * np.sqrt(K * F0) * (
                integral1_eval[0] + np.sin(eta * np.pi) * integral2_eval[0])

        return call_price

    def calc_iv(self, alpha, beta, rho, v, contract_params, T):
        K, F0 = contract_params

        # if K \approx f, the approximation is undefined. We take two points around the considered strike and use
        # linear interpolation
        if np.round(K, 4) == np.round(F0, 4):
            offset = .006
            K2 = (1 + offset) * F0
            K1 = (1 - offset) * F0
            iv1 = self.calc_iv(alpha, beta, rho, v, (K1, F0), T)
            iv2 = self.calc_iv(alpha, beta, rho, v, (K2, F0), T)

            helling = (iv2 - iv1) / (K2 - K1)
            return helling * (K - K1) + iv1

        # calculate call price and its IV using Jaeckel (2015).
        call_price = self.calc_call(alpha, beta, rho, v, contract_params, T)
        r = 0
        iv = implied_volatility.implied_volatility(call_price, F0, K, r, T, 'c')
        return iv.iloc[0, 0]

class WrappedModel(nn.Module):
    """
    Architecture of the neural netwrok used for SABR approximation.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 340),
            nn.BatchNorm1d(340),
            nn.Softplus(),

            nn.Linear(340, 260),
            nn.BatchNorm1d(260),
            nn.Softplus(),

            nn.Linear(260, 440),
            nn.BatchNorm1d(440),
            nn.Softplus(),

            nn.Linear(440, 380),
            nn.BatchNorm1d(380),
            nn.Softplus(),

            nn.Linear(380, 190),
            nn.BatchNorm1d(190),
            nn.Softplus(),

            nn.Linear(190, 1),
        )

    def forward(self, x):
        return self.net(x)


class NeuralNetwork:
    """
    Neural network SABR approximation trained on Monte Carlo generated training dataset.
    """
    def __init__(self):
        # Load model weights and scalers
        model_path = "functions/nn model/best_model.pth"
        scaler_X_path = "functions/nn model/scaler_X.pkl"
        scaler_Y_path = "functions/nn model/scaler_Y.pkl"

        # Build model and load trained weights
        self.model = WrappedModel()
        state_dict = torch.load(model_path, map_location="cpu")

        # Fix possible prefix mismatch (since model saved as Sequential)
        if not any(k.startswith("net.") for k in state_dict.keys()):
            state_dict = {f"net.{k}": v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        # Load input/output scalers
        self.scaler_X = joblib.load(scaler_X_path)
        self.scaler_Y = joblib.load(scaler_Y_path)

    def get_name(self):
        return "Neural Network"

    def calc_iv(self, alpha, beta, rho, v, contract_params, T):

        K, F = contract_params

        # Match your training feature engineering
        error = F - 0.05
        F = 0.05
        K = (F * K) / (F + error)
        alpha = alpha * ((F / (F + error)) ** beta)

        relative_strike = K / F

        # Prepare and scale input
        X_input = np.array([[alpha, rho, v, T, relative_strike]])
        X_scaled = self.scaler_X.transform(X_input)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Predict (no grad)
        with torch.no_grad():
            iv_scaled = self.model(X_tensor).numpy()
        iv = self.scaler_Y.inverse_transform(iv_scaled)

        return float(iv[0, 0])

class Hagan():
    """
    Hagan et al. (2002) implied volatility approximation
    Equations (3.12-16)
    """

    def calc_iv(self, alpha, beta, rho, nu, contract_params, T):
        K, f = contract_params
        lf = np.log(f / K)

        if f == K:
            return (alpha / f**(1 - beta)) * (
                1 + T * (
                    ((1 - beta)**2 / 24) * (alpha**2 / f**(2 - 2*beta))
                    + 0.25 * rho * beta * nu * alpha / f**(1 - beta)
                    + (2 - 3*rho*rho) * nu*nu / 24
                )
            )

        z = (nu / alpha) * (f*K)**(0.5*(1 - beta)) * lf
        xz = np.log((np.sqrt(1 - 2*rho*z + z*z) + z - rho) / (1 - rho))

        I0 = (alpha * z) / (
            (f*K)**(0.5*(1 - beta)) *
            (1 + (1 - beta)**2 * lf*lf / 24 + (1 - beta)**4 * lf**4 / 1920) *
            xz
        )

        I1 = (
            ((1 - beta)**2 / 24) * (alpha**2 / (f*K)**(1 - beta))
            + 0.25 * rho * beta * nu * alpha / (f*K)**(0.5*(1 - beta))
            + (2 - 3*rho*rho) * nu*nu / 24
        )

        return I0 * (1 + I1 * T)

    def calc_call(self, alpha, beta, rho, nu, contract_params, T):
        return np.nan

    def get_name(self):
        return 'Hagan'

class Obloj():
    """
    Obloj (2008) implied volatility approximation.
    Equations (3.12), (3.14), (3.20-22)
    """

    def calc_iv(self, alpha, beta, rho, v, contract_params, T):
        K, F0 = contract_params

        x = np.log(F0 / K)
        z = (v / alpha) * ((F0**(1-beta)) - (K**(1-beta))) / (1-beta)
        x_z = np.log(((np.sqrt(1 - 2 * rho * z + z**2)) + z - rho) / (1 - rho))

        if K != F0:
            I0 = v * x / x_z
        else:
            I0 = alpha * (K ** (beta - 1))

        I1 = ((beta - 1)**2 / 24) * ((alpha**2) / ((F0 * K)**(1 - beta))) + (1/4) * (rho * v * alpha * beta) / ((F0 * K)**((1 - beta)/2)) + (2 - 3 * rho**2) / 24 * v**2

        iv = I0 * (1 + I1 * T)

        return iv

    def calc_call(self, alpha, beta, rho, v, contract_params, T):
        return np.nan

    def get_name(self):
        return 'Obloj'

class MonteCarlo():
    """
    Monte Carlo-based implied volatility approximation using Euler discretization.
    Requires CuPy
    """

    def simulate_paths(self, no_of_sim, no_of_steps, expiry, F0, alpha, beta, rho, nu, seed=None):
        # Time increments
        dt = expiry / no_of_steps
        dt_sqrt = cp.sqrt(dt)

        # Seed
        rng = cp.random.RandomState(seed) if seed is not None else cp.random

        # Batching to prevent GPU memory overflow
        chunk_size = 500000
        final_results = []

        # Loop over the total number of simulations in batches
        for chunk_start in range(0, no_of_sim, chunk_size):
            chunk_end = min(chunk_start + chunk_size, no_of_sim)
            current_chunk_size = chunk_end - chunk_start

            # Initialize forward prices and volatilities
            F0 = cp.full(current_chunk_size, F0, dtype=cp.float32)
            alpha = cp.full(current_chunk_size, alpha, dtype=cp.float32)

            # Create array of 1s to indicate alive status
            alive = cp.ones(current_chunk_size, dtype=cp.bool_)

            for t in range(no_of_steps):
                
                # Generate standard normal random numbers for each path
                Z1 = rng.normal(size=current_chunk_size, dtype=cp.float32)
                Y1 = rng.normal(size=current_chunk_size, dtype=cp.float32)

                dW_F = Z1
                dW_alpha = rho * Z1 + cp.sqrt(1 - rho**2) * Y1

                # Updating forward and volatility for alive paths
                F_b = cp.where(alive, F0 ** beta, 0.0)
                F_new = F0 + alpha * F_b * dW_F * dt_sqrt
                alpha_new = alpha + nu * alpha * dW_alpha * dt_sqrt

                F0 = cp.where(alive, F_new, F0)
                alpha = cp.where(alive, alpha_new, alpha)
                
                # Identify and absorb paths that hit zero
                dead = (F0 <= 0)
                F0 = cp.where(dead, 0.0, F0)
                alpha = cp.where(dead, 0.0, alpha)
                alive = alive & ~dead

                if not cp.any(alive):
                    break

            final_results.append(F0.get())

        # Combine results from all batches
        return np.concatenate(final_results)

    def calc_price_and_iv(self, alpha, beta, rho, nu, K_vector, F0, T, no_of_sim, no_of_steps, seed=None):
        F_T = self.simulate_paths(no_of_sim, no_of_steps, T, F0, alpha, beta, rho, nu, seed=seed)
        call_prices = np.array([np.mean(np.maximum(F_T - K, 0.0)) for K in K_vector])
        call_ivs = vectorized_implied_volatility(call_prices, F0, K_vector, T, r=0, flag='c', q=0, return_as='numpy', on_error='ignore')
        return call_prices, call_ivs
