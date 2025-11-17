## Abstract
This thesis investigates the use of a feedforward neural network as an approximation method for the implied volatility map of the SABR model, with a particular focus on swaption markets. The objective is to identify an approach that improves upon traditional approximations regarding both computational efficiency and accuracy. Using a two-step pointwise approach and a large synthetic training dataset, generated using Monte Carlo simulation, the neural network learns the mapping from SABR model parameters to implied volatilities. It is then evaluated on another synthetic dataset of swaptions, regarding its accuracy in generating implied volatilities from known SABR parameters, calibrating full volatility curves to recover the underlying SABR parameters, and producing option Greeks. Across nearly all evaluation metrics, the neural network approximation outperforms both the formula of Hagan et al. (2002) and the Zero Correlation Map of Antonov et al. (2013). Its principal shortcoming, however, lies in a higher number of arbitrage violations compared to other approximation methods.


## Development

<p align="center">

### Core Libraries
![numpy](https://img.shields.io/badge/numpy-1.26.4-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.1.4-150458?logo=pandas&logoColor=white)
![scipy](https://img.shields.io/badge/scipy-1.11.4-0C55A5?logo=scipy&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14.0-006699)
![joblib](https://img.shields.io/badge/joblib-1.2.0-9cf)
![tqdm](https://img.shields.io/badge/tqdm-4.65.0-333333)

### Machine Learning
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-F7931E?logo=scikitlearn&logoColor=white)
![torch](https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C?logo=pytorch&logoColor=white)
![optuna](https://img.shields.io/badge/optuna-4.4.0-3f8cff)

### Visualization
![matplotlib](https://img.shields.io/badge/matplotlib-3.8.0-11557c?logo=matplotlib&logoColor=white)
![seaborn](https://img.shields.io/badge/seaborn-0.12.2-4C8CB5)
![plotly](https://img.shields.io/badge/plotly-5.9.0-3F4F75?logo=plotly&logoColor=white)

### Option Pricing
![py_vollib](https://img.shields.io/badge/py__vollib-1.0.1-lightgrey)
![py_vollib_vectorized](https://img.shields.io/badge/py__vollib__vectorized-0.1.1-lightgrey)

### GPU Acceleration
![cupy](https://img.shields.io/badge/cupy-latest-00A37A?logo=nvidia&logoColor=white)

</p>
