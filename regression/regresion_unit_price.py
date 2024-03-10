import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional, Union
import copy
import os

from xgboost import DMatrix, train, XGBRegressor
from model_regression import regressionUnitPrice
from utils_regression import data_cleaning, recency, frequency_monetary, outlier_removal, feature_scaling, cancelled_transaction
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import logging

# Ignore warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

np.random.seed(123)

logging.basicConfig(filename="regression.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 
logger=logging.getLogger()

logger.setLevel(logging.INFO)

def xg_boost(x_train : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]], x_test : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]], 
             y_train : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]], y_test : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]]):
    r"""Function to train and evaluate xg_boost

        Args:
            x_train (pd.DataFrame): Input Dataframe object (train)
            x_test (pd.DataFrame): Input Dataframe object (test)
            y_train (pd.DataFrame): Input Dataframe object (train)
            y_test (pd.DataFrame): Input Dataframe object (test)

        Returns:
            return_type: prediction
    """
    try:
        # parameters for learning and model instansiation
        model = XGBRegressor(objective='reg:squarederror', n_estimators=500, max_depth=7, eta=0.01, subsample=0.7, colsample_bytree=0.8)
        
        # Fit the model on train data
        model.fit(x_train, y_train)
        # Evaluate the model on test data
        predictions = model.predict(x_test)

        # Calculate MSE
        mse = mean_squared_error(y_test, predictions)
        logger.info(f'Mean Squared Error of XG_Boost on test data: {mse}\n')

        # Calculate RMSE
        rmse = mean_squared_error(y_test, predictions, squared=False)
        logger.info(f'Root Mean Squared Error of XG_Boost on test data: {rmse}\n')
        
        return predictions
    
    except Exception as e:
        logger.info("Error while running the XG_Boost regression function")
        logger.exception("Error in xg_boost function " + str(e))
        
def neural_network(x_train : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]], x_test : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]], 
             y_train : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]], y_test : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]]):
    r"""Function to train and evaluate neural network

        Args:
            x_train (pd.DataFrame): Input Dataframe object (train)
            x_test (pd.DataFrame): Input Dataframe object (test)
            y_train (pd.DataFrame): Input Dataframe object (train)
            y_test (pd.DataFrame): Input Dataframe object (test)

        Returns:
            return_type: prediction
    """
    try: 
        # Instantiate the model
        model = regressionUnitPrice()
        
        # Convert the data into tensors
        x_train = torch.tensor(x_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
        x_test = torch.tensor(x_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)
            
        # loss function and optimizer
        loss_fn = nn.MSELoss()  # mean square error
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        
        n_epochs = 200   # number of epochs to run
        batch_size = 128  # size of each batch
        batch_start = torch.arange(0, len(x_train), batch_size)
        
        # Hold the best model
        best_mse = np.inf   # init to infinity
        best_weights = None
        history = []
        
        for epoch in range(n_epochs):
            model.train()
            with tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    # take a batch
                    x_batch = x_train[start:start+batch_size]
                    y_batch = y_train[start:start+batch_size]
                    # forward pass
                    y_pred = model(x_batch)
                    loss = loss_fn(y_pred, y_batch)
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()
                    # print progress
                    bar.set_postfix(mse=float(loss))
            # evaluate accuracy at end of each epoch
            model.eval()
            y_pred = model(x_test)
            mse = loss_fn(y_pred, y_test)
            mse = float(mse)
            history.append(mse)
            if mse < best_mse:
                best_mse = mse
                best_weights = copy.deepcopy(model.state_dict())
        
        return history, best_mse, torch.squeeze(y_pred), best_weights
    
    except Exception as e:
        logger.info("Error while running the NN regression function")
        logger.exception("Error in neural_network function " + str(e))
        
if __name__ == '__main__': 
    
    # Load the input *.csv file as a Pandas DataFrame
    df = pd.read_csv("./app/data.csv", encoding="ISO-8859-1")
    
    # Log all the details of the input data
    shape_init = df.shape
    logger.info('\n')
    logger.info('\n' + '#'*20 + 'RUNNING THE REGRESSION SCRIPT' + '#'*20 + '\n')
    logger.info('#'*20 + 'SHAPE OF THE INPUT DATA' + '#'*20)
    logger.info(f'Total number of rows and columns in the initial dataset is : {shape_init[0]} and {shape_init[1]} respectively\n\n')
    logger.info('#'*20 + 'STATISTICS OF THE INPUT DATA' + '#'*20)
    # Statistical summary for numerical variables
    logger.info(f'Statistics of the numerical variables is :\n{df.describe().T}\n\n')
    # Statistical summary for categorical variables
    logger.info(f'Statistics of the categorical variables is :\n{df.describe(include="object").T}\n\n')
    
    # Cleaned df
    df = data_cleaning(df)
    df = cancelled_transaction(df)
    
    # Log all the details after cleaning the data
    shape_clean = df.shape
    perc_removed = ((shape_init[0] - shape_clean[0]) / shape_init[0]) * 100
    logger.info('#'*20 + 'DATA AFTER CLEANING' + '#'*20)
    logger.info(f'Total number of rows and columns in the initial dataset is : {shape_clean[0]} and {shape_clean[1]} respectively, after dropping {perc_removed}% of the rows\n\n')
    logger.info(f'The total unique items are {df["StockCode"].nunique()} and unique customers are {df["CustomerID"].nunique()}\n\n')
    
    # Feature Engineering
    item_df = recency(df)
    item_df = frequency_monetary(df, item_df)
    
    # Outlier removal and feature scaling
    items_df_no_outliers, items_df_outliers = outlier_removal(item_df)
    items_df_scaled = feature_scaling(items_df_no_outliers)
    
    x = items_df_scaled.drop('UnitPrice', axis=1)
    y = items_df_scaled['UnitPrice']    
    
    # Train test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
    
    dir_results = os.path.join(os.getcwd(), 'results_regression')
    os.makedirs(dir_results, exist_ok=True)
        
    # prediction of 'UnitPrice' using XG_Boost model
    pred_xgb = xg_boost(x_train, x_test, y_train, y_test)
    
    # prediction of 'UnitPrice' using XG_Boost model using NN
    mse_history, best_mse, pred_nn, best_weights = neural_network(x_train, x_test, y_train, y_test)

        
    # Plotting the results of NN method
    logger.info(f'Mean Squared Error of NN on test data: {best_mse}\n')
    logger.info(f"Root Mean Squared Error of NN on test data: {np.sqrt(best_mse)}")
    plt.figure(figsize=(10, 5))
    plt.plot(mse_history)
    plt.xlabel('Epoch')
    plt.ylabel('MSE_eval')
    fig_name_nn = os.path.join(dir_results, 'MSE_eval_NN' + '.' + 'png')
    plt.savefig(fig_name_nn)
    
    # Residuals for plotting residual frequency
    pred_nn_np = pred_nn.cpu().detach().numpy()
    residuals_xgb = y_test - pred_xgb
    residuals_nn = y_test - pred_nn_np
    
    plt.figure(figsize=(10, 5))
    plt.hist([residuals_xgb], alpha=0.5, label='XGB', bins=20)
    plt.hist([residuals_nn], alpha=0.5, label='NN', bins=20)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.legend(loc='upper right')
    fig_name_res = os.path.join(dir_results, 'residual_XGB_NN' + '.' + 'png')
    plt.savefig(fig_name_res) 
    
    
    
    