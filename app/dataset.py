from torch.utils.data import Dataset
import torch
import numpy as np
from typing import Optional, Union
import pandas as pd
import logging

logging.basicConfig(filename="train_eval.log", 
					format='%(asctime)s %(message)s', 
					filemode='a') 
logger=logging.getLogger()

class eCommerceDataset(Dataset):
    """eCommerceDataset PyTorch Dataset for training
    
    Args:
        df_train (pd.DataFrame): Dataframe containing the "StockCode"/"CustomerID" for training
        all_itemIds (array_type): List containing all unique "StockCode"
    
    """
    def __init__(self, df_train : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]], all_itemIds : Optional[Union[np.ndarray, list]]):
        self.users, self.items, self.labels = self.get_dataset(df_train, all_itemIds)

    def __len__(self):
        return len(self.users)
  
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, df_train : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]], all_itemIds : Optional[Union[np.ndarray, list]]):
        try:
            users, items, labels = [], [], []
            user_item_set = set(zip(df_train['CustomerID'], df_train['StockCode']))

            # For each user-item interaction, 4 negetive samples are created (no user-item interaction)
            num_negatives = 4
            for u, i in user_item_set:
                users.append(u)
                items.append(i)
                labels.append(1)
                for _ in range(num_negatives):
                    negative_item = np.random.choice(all_itemIds)
                    while (u, negative_item) in user_item_set:
                        negative_item = np.random.choice(all_itemIds)
                    users.append(u)
                    items.append(negative_item)
                    labels.append(0)

            return torch.tensor(users), torch.tensor(items), torch.tensor(labels)
        
        except Exception as e:
            logger.info("Error in converting the data to tensors for the model")
            logger.exception("Error in get_dataset function " + str(e))