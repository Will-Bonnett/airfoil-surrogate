import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from mlp import fit_predict_mlp_single
# Mean floor
def fit_predict_mean(x,y, test):
    """
    Predicts cl and cd using the mean of the training data

    inputs:
        x - Training data features for model to be trained off
        y - Training data cl and cd
        test - Testing data features
    """

    # Calculate mean
    cl_mean = y["cl"].mean()
    cd_mean = y["cd"].mean()

    # Predict with mean
    cl_pred = np.full(len(test),cl_mean)
    cd_pred = np.full(len(test),cd_mean)

    return {
        "cl": cl_pred,
        "cd": cd_pred
        }

def fit_predict_thin_airfoil(x,y,test):
    """
    Predicts cl using the Thin Airfoil Theory

    inputs:
        x - Training data features for model to be trained off
        y - Training data cl and cd
        test - Testing data features
    """
    cl_pred = 2*np.pi * np.deg2rad(test["angleOfAttack"])

    return {"cl":np.asarray(cl_pred), "cd":None}

def fit_predict_linear(x,y,test):
    """
    Predicts cl and cd using Linear Regression

    inputs:
        x - Training data features for model to be trained off
        y - Training data cl and cd
        test - Testing data features
    """
    cl_model = LinearRegression().fit(x,y["cl"])
    cd_model = LinearRegression().fit(x,y["cd"])
    cl_pred = cl_model.predict(test)
    cd_pred = cd_model.predict(test)

    return {
        "cl": cl_pred,
        "cd": cd_pred
        }

def fit_predict_gb(x,y,test):
    """
    Predicts cl and cd using Gradient Boosing

    inputs:
        x - Training data features for model to be trained off
        y - Training data cl and cd
        test - Testing data features
    """

    cl_model = HistGradientBoostingRegressor(random_state=0).fit(x,y["cl"])
    cd_model = HistGradientBoostingRegressor(random_state=0).fit(x,y["cd"])
    cl_pred = cl_model.predict(test)
    cd_pred = cd_model.predict(test)    

    return {
        "cl": cl_pred,
        "cd": cd_pred
        }

def fit_predict_mlp(x,y,test):
    """
    Predicts cl and cd using multi-layer perceptrons

    inputs:
        x - Training data features for model to be trained off
        y - Training data cl and cd
        test - Testing data features
    """
    SEEDS = np.arange(10,300,100)
    
    cl_pred = np.mean([fit_predict_mlp_single(x, y[["cl"]], test, seed=s) for s in SEEDS], axis=0)
    cd_pred = np.mean([fit_predict_mlp_single(x, y[["cd"]], test, seed=s) for s in SEEDS], axis=0)

    return {
        "cl": cl_pred,
        "cd": cd_pred
        }