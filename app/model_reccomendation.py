import torch.nn as nn
import torch
import pytorch_lightning as pl  
from torch.utils.data import DataLoader
from typing import Optional, Union
import logging
import numpy as np
import pandas as pd
import os

arg_value = os.environ.get("SCRIPT_MODE")

if arg_value == "train_reccomendation":
    from dataset import eCommerceDataset
else:
    from app.dataset import eCommerceDataset
    # from dataset import eCommerceDataset

logging.basicConfig(filename="train_eval.log", 
					format='%(asctime)s %(message)s', 
					filemode='a') 
logger=logging.getLogger()


class NCF(pl.LightningModule):
    """ Neural Collaborative Filtering (NCF)
    
        Args:
            num_users (int): Number of unique users
            num_items (int): Number of unique items
            df_train (pd.DataFrame): Dataframe containing the "StockCode"/"CustomerID" for training
            all_itemIds (array_type): List containing all unique "StockCode" (train + test)
    """
    
    def __init__(self, num_users : Optional[Union[int, float]], num_items : Optional[Union[int, float]], 
                 df_train : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]], all_itemIds : Optional[Union[np.ndarray, list]]):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=128)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=128)
        self.fc1 = nn.Linear(in_features=256, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=128)
        self.fc5 = nn.Linear(in_features=128, out_features=64)
        self.fc6 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        self.df_train = df_train
        self.all_itemIds = all_itemIds
        
    def forward(self, user_input : Optional[Union[int, float]], item_input : Optional[Union[int, float]]):
        try:
            # Pass through embedding layers
            user_embedded = self.user_embedding(user_input)
            item_embedded = self.item_embedding(item_input)

            # Concat the two embedding layers
            vector = torch.cat([user_embedded, item_embedded], dim=-1)
            # Pass through dense layer
            vector = nn.ReLU()(self.fc1(vector))
            vector = nn.ReLU()(self.fc2(vector))
            vector = nn.ReLU()(self.fc3(vector))
            vector = nn.ReLU()(self.fc4(vector))
            vector = nn.ReLU()(self.fc5(vector))
            vector = nn.ReLU()(self.fc6(vector))
            # Output layer
            pred = nn.Sigmoid()(self.output(vector))

            return user_embedded, item_embedded, vector, pred
        
        except Exception as e:
            logger.info("Error while running the forward pass of the model")
            logger.exception("Error in forward function " + str(e))
    
    def training_step(self, batch, batch_idx):
        try:
            user_input, item_input, labels = batch
            user_embedding, item_embedding, joint_embedding, predicted_labels = self(user_input, item_input)
            loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return loss
                
        except Exception as e:
            logger.info("Error while running the training step")
            logger.exception("Error in training_step function " + str(e))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(eCommerceDataset(self.df_train, self.all_itemIds),
                                                batch_size=128, num_workers=4, shuffle=True)