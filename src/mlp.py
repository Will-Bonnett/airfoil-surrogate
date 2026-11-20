import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from evaluation import add_results


def prepare_data(train_df, test_df, features, targets):
    # random_state remains 0 to ensure every model is trained on the exact same data
    x_t, x_v, y_t, y_v = train_test_split(train_df[features], train_df[targets], test_size=0.2, random_state=0)

    # Get scalars:
    x_scalar = StandardScaler().fit(x_t)
    y_scalar = StandardScaler().fit(y_t)

    # Apply Scalars:
    x_train = torch.tensor(x_scalar.transform(x_t),dtype=torch.float32)
    x_val = torch.tensor(x_scalar.transform(x_v),dtype=torch.float32)
    x_test = torch.tensor(x_scalar.transform(test_df[features]),dtype=torch.float32)
    y_train = torch.tensor(y_scalar.transform(y_t),dtype=torch.float32)
    y_val = torch.tensor(y_scalar.transform(y_v),dtype=torch.float32)
    y_test = torch.tensor(y_scalar.transform(test_df[targets]),dtype=torch.float32)

    return {
        "x_train":x_train,
        "y_train":y_train,
        "x_test":x_test,
        "y_test":y_test,
        "x_val":x_val,
        "y_val":y_val,
        "x_scalar":x_scalar,
        "y_scalar":y_scalar
    }

def prepare(features, targets, test):
    # random_state remains 0 to ensure every model is trained on the exact same data
    x_t, x_v, y_t, y_v = train_test_split(features, targets, test_size=0.2, random_state=0)

    # Get scalars:
    x_scalar = StandardScaler().fit(x_t)
    y_scalar = StandardScaler().fit(y_t)

    # Apply Scalars:
    x_train = torch.tensor(x_scalar.transform(x_t),dtype=torch.float32)
    x_val = torch.tensor(x_scalar.transform(x_v),dtype=torch.float32)
    x_test = torch.tensor(x_scalar.transform(test),dtype=torch.float32)
    y_train = torch.tensor(y_scalar.transform(y_t),dtype=torch.float32)
    y_val = torch.tensor(y_scalar.transform(y_v),dtype=torch.float32)

    return {
        "x_train":x_train,
        "y_train":y_train,
        "x_test":x_test,
        "x_val":x_val,
        "y_val":y_val,
        "x_scalar":x_scalar,
        "y_scalar":y_scalar
    }

def build_model(n_inputs, n_outputs, hidden=64):
    model = nn.Sequential(
    nn.Linear(n_inputs,hidden),nn.ReLU(),
    nn.Linear(hidden,hidden),nn.ReLU(),
    nn.Linear(hidden,n_outputs))
    
    return model

def train_model(model,data,epochs,lr,batch_size):

    # Parameters 
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(),lr=lr)

    batches = DataLoader(
        TensorDataset(data["x_train"],data["y_train"]),
        batch_size=batch_size,
        shuffle=True)


    train_losses = []
    val_losses = []
    # Training Loop
    for epoch in range(epochs):
        batch_losses=[]
        for x_batch, y_batch in batches:
            # Predict from x
            pred = model(x_batch)
            # measure error
            loss = loss_fn(pred,y_batch)
            # clear previous gradients
            optimiser.zero_grad()
            # backpropagation
            loss.backward()
            # nudge the weights
            optimiser.step()

            batch_losses.append(loss.item())
        # record training and validation loss
        train_losses.append(np.mean(batch_losses))

        model.eval()
        with torch.no_grad():
            val_losses.append(loss_fn(model(data["x_val"]),data["y_val"]).item())
        model.train()

    
    return train_losses, val_losses

def predict(model,x_test,y_scalar):
    model.eval()
    with torch.no_grad():
        pred = model(x_test)
    pred = y_scalar.inverse_transform(pred)

    return pred

def run_experiment(features, targets,train_df, test_df,results,model_name,seed=0,batch_size = 32,num_epochs = 100,learning_rate = 0.001):
    torch.manual_seed(seed)

    # prepare data
    data = prepare_data(train_df, test_df, features, targets)
    # build model
    model = build_model(len(features), len(targets))
    # train the model
    train_l, val_l = train_model(model, data, num_epochs, learning_rate, batch_size)
    # predict from model
    pred = predict(model,data["x_test"],data["y_scalar"])
    # evaluate model

    # pred has one column per target, in the same order as `targets`
    kwargs = {}
    for i, target in enumerate(targets):
        kwargs[f"{target}_true"] = test_df[target]
        kwargs[f"{target}_pred"] = pred[:, i]
    results = add_results(results, model_name, **kwargs)
    
    # return results, predicted values, train_1osses, val_losses
    return results, pred, train_l, val_l
    

def fit_predict_mlp_single(x,y,test,seed=0,batch_size = 32,num_epochs = 100,learning_rate = 0.001):
    torch.manual_seed(seed)

    data = prepare(x,y,test)
    model = build_model(len(x.columns),1)
    train_l, val_l = train_model(model, data, num_epochs, learning_rate, batch_size)
    pred = predict(model,data["x_test"],data["y_scalar"])

    return pred[:,0]
    