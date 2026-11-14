from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def add_results(results, model, cl_true=None, cl_pred=None, cd_true=None, cd_pred=None):
    """
    Scores and Saves a models evaluation results to the given dict

    Args
        - results   previous results table (dict), updated in place and returned
        - model     String of model name for results to be saved under
        - cl_true   (optional) True cl values
        - cl_pred   (optional) Predicted cl values
        - cd_true   (optional) True cd values
        - cd_pred   (optional) Predicted cd values
        A target is only scored if its prediction is given; pass the true
        values and predictions for a target together, by name, e.g.
        add_results(results, "rf", cl_true=y_true, cl_pred=y_pred)
    Returns
        - results   returns table
    """
    row = {}

    # get Scores
    if cl_pred is not None:
        cl_mae, cl_rmse, cl_r2 = evaluate(cl_true,cl_pred)
        row["cl_mae"] = cl_mae
        row["cl_rmse"] = cl_rmse
        row["cl_r2"] = cl_r2

        print("\n--- Cl (coefficient of Lift) ---")
        print(f"MAE: {cl_mae}")
        print(f"RMSE: {cl_rmse}")
        print(f"R^2: {cl_r2}")

    if cd_pred is not None:
        cd_mae, cd_rmse, cd_r2 = evaluate(cd_true,cd_pred)
        row["cd_mae"] = cd_mae
        row["cd_rmse"] = cd_rmse
        row["cd_r2"] = cd_r2

        print("--- Cd (coefficient of Drag) ---")
        print(f"MAE: {cd_mae}")
        print(f"RMSE: {cd_rmse}")
        print(f"R^2: {cd_r2}")

    results[model] = row
    return results


def evaluate(y_true,y_pred):
    """
    Calculates scores of the evaluation results

    Args
        - y_true    True values
        - y_pred    Predicted values
    Returns
        - mae   Mean Absolute Error
        - rmse  Root Mean Squared Error
        - r2    R^2 Score
    """
    mae = mean_absolute_error(y_true,y_pred)
    rmse = root_mean_squared_error(y_true,y_pred)
    r2 = r2_score(y_true,y_pred)
    return mae, rmse, r2


def plot_pred_vs_true(ax, true, pred, title):
    """
    Plot the true (x) vs predicted values (y)

    Args
        - ax    graph to be added to
        - true  True values 
        - pred  Predicted values
        - title Title of graph
    """
    ax.scatter(true,pred)

    # Y=X line
    lims = [np.min([ax.get_xlim(),ax.get_ylim()]),np.max([ax.get_xlim(),ax.get_ylim()])]
    ax.plot(lims, lims,c="red")

    ax.set_title(title)
    ax.set_ylabel("Predicted")
    ax.set_xlabel("True")
