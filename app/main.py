import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from app.model_reccomendation import NCF
import _pickle as cPickle

from typing import Union
from fastapi import FastAPI
import json
import logging
from typing import Optional, Union

# Ignore warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(filename="test.log", 
					format='%(asctime)s %(message)s', 
					filemode='a') 
logger=logging.getLogger()

app = FastAPI()

def values_labels(input_dict : Optional[Union[dict, str]]):
    r"""Function used to convert given input "CustomerID" and "StockCode" to labels

        Args:
            input_dict (dict): Input dictionary containing "CustomerID" and "StockCode"

        Returns:
            return_type: dict
    """
    try:
        with open('./pickle_dump/items_to_labels.pkl', 'rb') as handle:
            items_to_labels = cPickle.load(handle)
            
        with open('./pickle_dump/users_to_labels.pkl', 'rb') as handle:
            users_to_labels = cPickle.load(handle)
            
        input_dict = json.loads(input_dict)
        
        for key, value in input_dict.items():
            
            if key == 'CustomerID':
                for i in range(len(value)):
                    if value[i] in users_to_labels:
                        value[i] = users_to_labels[value[i]]
                    else:
                        print('provide valid CustomerID')
                
            elif key == 'StockCode':
                for i in range(len(value)):
                    if value[i] in items_to_labels:
                        value[i] = items_to_labels[value[i]]
                    else:
                        print('provide valid StockCode')

        return input_dict
    
    except Exception as e:
        logger.info("Error while converting the CustomerID's and StockCode's to their labels during deployment")
        logger.exception("Error in values_labels function " + str(e))

def test(input_example : Optional[Union[dict, str]], df : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]], 
         all_items_labels : Optional[Union[np.ndarray, list]], model : Optional[NCF]):
    r"""Function used to return the top-k relevant object for a given "CustomerID"/"StockCode" pair as suggested by the trained model

        Args:
            input_example (dict): Input dictionary containing "CustomerID" and "StockCode"
            df (pd.DataFrame): Input Dataframe object (train + test)
            all_items_labels (array_type): List containing all unique "StockCode" as labels(train + test)
            model (NCF) : Trained model for evaluating it's performance
        Returns:
            return_type: dict
    """
    try:
        model.to('cpu')
        model.eval()
        # Convert the CustomerID's and StockCode's to their respective labels for model to process
        input_example = values_labels(input_example)
        # User-item pairs for testing
        test_user_item_set = set(zip(input_example['CustomerID'], input_example['StockCode']))
        # Dict of all items that are interacted with by each user
        user_interacted_items = df.groupby('CustomerID')['StockCode'].apply(list).to_dict()
        
        top_5_global = {}
        
        for (u,i) in tqdm(test_user_item_set):
            interacted_items = user_interacted_items[u]
            not_interacted_items = list(list(set(all_items_labels) - set(interacted_items)))
            test_items = not_interacted_items + [i]
            
            # Model predicted embeddings and labels
            user_embedding, item_embedding, joint_embedding, predicted_labels = model(torch.tensor([u]*len(test_items)), 
                                                torch.tensor(test_items))
            
            predicted_labels = np.squeeze(predicted_labels.detach().numpy())
            
            top5_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:5].tolist()]
            
            top_5_global[u] = top5_items
            
        return top_5_global
    
    except Exception as e:
        logger.info("Error while testing the model for returning top-k items")
        logger.exception("Error in test function " + str(e))

@app.get("/{input_example}")
def main(input_example):
    r"""Main functions which recieves the input through the FastAPI and send the response back as a json object

        Args:
            input_dict (dict): Input dictionary containing "CustomerID" and "StockCode"

        Returns:
            return_type: dict/json
    """
    try:

        # Example CustomerID's and their respective interacted StockCode's (before converted to their respective labels) in the test_df
        unpickled_df_cleaned = pd.read_pickle("./pickle_dump/df_cleaned.pkl")
        unpickled_df_test_cleaned = pd.read_pickle("./pickle_dump/df_test_cleaned.pkl")
        
        # Instantiate the model from the checkpoint
        all_items_labels = unpickled_df_cleaned['StockCode'].unique()
        num_users = unpickled_df_cleaned['CustomerID'].nunique()
        num_items = unpickled_df_cleaned['StockCode'].nunique()
        
        # Load the trained model from the checkpoint for reccomendation of top-k personalized item recommendation
        model = NCF.load_from_checkpoint("./model/epoch=99-step=515700.ckpt", num_users=num_users, num_items=num_items, 
                                        df_train=unpickled_df_cleaned, all_itemIds=all_items_labels)

        # Get the top-5 relevant StockCode's for each "CustomerID"/"StockCode" pair
        top_5 = test(input_example, unpickled_df_cleaned, all_items_labels, model)
    
        # Convert the returned top-5 labels to have the description along with the top-5 item ID's
        with open('./pickle_dump/labels_to_items.pkl', 'rb') as handle:
            labels_to_items = cPickle.load(handle)
            
        with open('./pickle_dump/labels_to_users.pkl', 'rb') as handle:
            labels_to_users = cPickle.load(handle)
        
        # Format the response in the required format
        final_dict = {}
        for user, items in top_5.items():
            dict_new = {}
            unit_price_list = []
            for item in items:
                description = unpickled_df_cleaned[unpickled_df_cleaned['StockCode'] == item]['Description'].values[0]
                unit_price = unpickled_df_cleaned[unpickled_df_cleaned['StockCode'] == item]['UnitPrice'].values[0]
                unit_price_list.append(unit_price)
                item = labels_to_items[item]
                dict_new[item] = description
            sum_bundle_item = sum(unit_price_list)
            bundle_price = {}
            bundle_price['sum_bundle_price'] = sum_bundle_item
            final_dict[labels_to_users[user]] = {'fetched_item_description' : dict_new, 'sum_bundle_price' : bundle_price}

        return final_dict
    
    except Exception as e:
        logger.info("Error while getting or sending the request from the main file")
        logger.exception("Error in main function " + str(e))
    
# if __name__ == '__main__': 
   
#    main()
    
    