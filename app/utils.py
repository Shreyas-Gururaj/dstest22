import _pickle as cPickle
import os 
import logging
import pandas as pd
import numpy as np
from typing import Optional, Union
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logging.basicConfig(filename="train_eval.log", 
					format='%(asctime)s %(message)s', 
					filemode='a') 
logger=logging.getLogger()

def data_cleaning(df : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]]):
    r"""Function used to clean the input dataframe

        Args:
            df (pd.DataFrame): Input Dataframe object (train + test)

        Returns:
            return_type: pd.DataFrame  
    """
    try:
        # Removing rows with missing values in 'CustomerID' and 'Description' columns 
        df = df.dropna(subset=['CustomerID', 'Description'])
        assert df.isnull().sum().sum() == 0, "There are still sum missing or NaN values in the df"
        
        # Remove the duplicate rows
        df.drop_duplicates(inplace=True)
        
        # Removing anamolies in the "StockCode" column
        unique_stock_codes = df['StockCode'].unique()
        anomalous_stock_codes = [code for code in unique_stock_codes if sum(c.isdigit() for c in str(code)) in (0, 1)]
        df = df[~df['StockCode'].isin(anomalous_stock_codes)]
        
        # Removing records with a unit price of zero to avoid potential data entry errors
        df = df[df['UnitPrice'] > 0]
        
        # Resetting the index of the cleaned dataset
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    except Exception as e:
            logger.info("Error in cleaning the df")
            logger.exception("Error in data_cleaning function " + str(e))
        
def dump_pickle(dir_pickle, objects_dict):
    """ Utility function to dump pickle objects
    
        Args:
            num_users (int): Number of unique users
            num_items (int): Number of unique items
            ratings (pd.DataFrame): Dataframe containing the movie ratings for training
            all_itemIds (array_type): List containing all movieIds (train + test)
    """
    try:
        os.makedirs(dir_pickle, exist_ok=True)
        suffix = 'pkl'
        
        for key, value in objects_dict.items():
            
            with open(os.path.join(dir_pickle, key + '.' + suffix), 'wb') as handle:
                cPickle.dump(value, handle)
    
    except Exception as e:
            logger.info("Error while converting the data to picke files")
            logger.exception("Error in dump_pickle function " + str(e))
        