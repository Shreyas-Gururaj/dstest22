import _pickle as cPickle
import os 
import logging
import pandas as pd
import numpy as np
from typing import Optional, Union
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logging.basicConfig(filename="regression.log", 
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
            
def cancelled_transaction(df : pd.DataFrame):
    r"""Function used to remove the cancelled transactions from the input dataframe only during unit price prediction task

        Args:
            df (pd.DataFrame): Input Dataframe object (train + test)

        Returns:
            return_type: pd.DataFrame  
    """
    try:
        #Filter out the rows with InvoiceNo starting with "C" and create a new column indicating the transaction status
        df['Transaction_Status'] = np.where(df['InvoiceNo'].astype(str).str.startswith('C'), 'Cancelled', 'Completed')

        # Only consider the 'Completed transactions for unit price prediction task
        df = df[df['Transaction_Status'] == 'Completed']
        # df.drop(['Transaction_Status'], axis=1)
        df.drop(columns=['Transaction_Status'], inplace=True)
        
        return df
        
    except Exception as e:
            logger.info("Error in removing the cancelled transactions the dataframe")
            logger.exception("Error in cancelled_transaction function " + str(e))
            
def recency(df : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]]):
    """ Function to convert the 'InvoiceDate' column to 'Days_Since_Last_Purchase'
    
        Args:
            df (pd.DataFrame): Input Dataframe object

        Returns:
            return_type: pd.DataFrame  
    """
    try:
        # Convert InvoiceDate to datetime type
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        # Convert InvoiceDate to datetime and extract only the date
        df['InvoiceDay'] = df['InvoiceDate'].dt.date
        # Find the most recent purchase date for each item
        item_data = df.groupby('StockCode')['InvoiceDay', 'UnitPrice'].max().reset_index()
        # Find the most recent date in the entire dataset
        most_recent_date = df['InvoiceDay'].max()

        # Convert InvoiceDay to datetime type before subtraction
        item_data['InvoiceDay'] = pd.to_datetime(item_data['InvoiceDay'])
        most_recent_date = pd.to_datetime(most_recent_date)

        # Calculate the number of days since the last purchase for each item
        item_data['Days_Since_Last_Purchase'] = (most_recent_date - item_data['InvoiceDay']).dt.days
        # Remove the InvoiceDay column
        item_data.drop(columns=['InvoiceDay'], inplace=True)
        
        return item_data
    
    except Exception as e:
            logger.info("Error while converting the date to days since last purchase")
            logger.exception("Error in recency function " + str(e))
            
def frequency_monetary(df : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]], df_item : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]]):
    """ Function to add the frequency and monetary values of purchase for each item in the 'StockCode' column
    
        Args:
            df (pd.DataFrame): Input Dataframe object 
            df_item (pd.DataFrame): Input Dataframe object
        Returns:
            return_type: pd.DataFrame  
    """
    try:
        # Calculate the total number of transactions made against each item
        total_transactions = df.groupby('StockCode')['InvoiceNo'].nunique().reset_index()
        total_transactions.rename(columns={'InvoiceNo': 'Total_Transactions'}, inplace=True)

        # Calculate the total number of times the product has been purchased
        total_products_purchased = df.groupby('StockCode')['Quantity'].sum().reset_index()
        total_products_purchased.rename(columns={'Quantity': 'Total_Products_Purchased'}, inplace=True)
                
        # Calculate the total spend against each item in total
        df['Total_Spend'] = df['UnitPrice'] * df['Quantity']
        total_spend = df.groupby('StockCode')['Total_Spend'].sum().reset_index()
        
        # Merge the new features into the df_item dataframe
        df_item = pd.merge(df_item, total_transactions, on='StockCode')
        df_item = pd.merge(df_item, total_products_purchased, on='StockCode')
        df_item = pd.merge(df_item, total_spend, on='StockCode')
        
        return df_item
    
    except Exception as e:
            logger.info("Error while adding the frequency, monetary values of each products purchased for each item in the StockCode")
            logger.exception("Error in frequency_monetary function " + str(e))

def outlier_removal(df : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]]):
    r"""Function used to remove outliers

        Args:
            df (pd.DataFrame): item_df Dataframe object

        Returns:
            return_type: pd.DataFrame  
    """
    try:
        # Initializing the IsolationForest model with a contamination parameter of 0.05
        model = IsolationForest(contamination=0.05, random_state=0)

        # Fitting the model on our dataset (converting DataFrame to NumPy to avoid warning)
        df['Outlier_Scores'] = model.fit_predict(df.iloc[:, 1:].to_numpy())

        # Creating a new column to identify outliers (1 for inliers and -1 for outliers)
        df['Is_Outlier'] = [1 if x == -1 else 0 for x in df['Outlier_Scores']]
        
        # Separate the outliers for analysis
        df_outlier = df[df['Is_Outlier'] == 1]
        
        # Remove the outliers from the main dataset
        df_cleaned = df[df['Is_Outlier'] == 0]

        # Drop the 'Outlier_Scores' and 'Is_Outlier' columns
        df_cleaned = df_cleaned.drop(columns=['Outlier_Scores', 'Is_Outlier'])
        
        # Reset the index of the cleaned data
        df_cleaned.reset_index(drop=True, inplace=True)
        
        return df_cleaned, df_outlier

    except Exception as e:
            logger.info("Error while removing outliers")
            logger.exception("Error in outlier_removal function " + str(e))
            
def feature_scaling(df : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]]):
    r"""Function used to normalize all independent variable

        Args:
            df (pd.DataFrame): item_df Dataframe object

        Returns:
            return_type: pd.DataFrame  
    """
    try:
        # Initialize the StandardScaler
        scaler = StandardScaler()
        
        # List of columns that don't need to be scaled
        columns_to_exclude = ['UnitPrice', 'StockCode']
        
        # List of columns that need to be scaled
        columns_to_scale = df.columns.difference(columns_to_exclude)
    
        # Copy the cleaned dataset
        df_scaled = df.copy()
        
        # Applying the scaler to the necessary columns in the dataset
        df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])

        # Setting CustomerID as the index column
        df_scaled.set_index('StockCode', inplace=True)

        return df_scaled

    except Exception as e:
            logger.info("Error while feature scaling")
            logger.exception("Error in feature_scaling function " + str(e))