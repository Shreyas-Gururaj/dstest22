import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from model_reccomendation import NCF


from utils import dump_pickle, data_cleaning
from sklearn.utils import shuffle
import argparse
from lightning.pytorch.loggers import WandbLogger, CSVLogger
import logging

# Ignore warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

np.random.seed(123)

# Set up the logging modules
# wandb_logger = WandbLogger(project="NeuralCF_Conrad", reinit=True)
csv_logger = CSVLogger("train_logs", name="NCF_train")

logging.basicConfig(filename="train_eval.log", 
					format='%(asctime)s %(message)s', 
					filemode='a') 
logger=logging.getLogger()

logger.setLevel(logging.INFO)

def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('training_recommendation')
    parser.add_argument('--accelerator', default='gpu', help='use cpu/gpu mode')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    return parser.parse_args()

def eval(df : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]], df_test : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]], 
         all_items_labels : Optional[Union[np.ndarray, list]], model : Optional[NCF]):
    r"""Function used to evaluate the model performance on the test dataset using the metric HR@10(HitRatio@10)

        Args:
            df (pd.DataFrame): Input Dataframe object (train + test)
            df_test (pd.DataFrame): Test Dataframe object
            all_items_labels (array_type): List containing all unique "StockCode" as labels(train + test)
            model (NCF) : Trained model for evaluating it's performance

        Returns:
            return_type: np.ndarray
    """
    try:
        # User-item pairs for testing
        test_user_item_set = set(zip(df_test['CustomerID'], df_test['StockCode']))

        # Dict of all items that are interacted with by each user
        user_interacted_items = df.groupby('CustomerID')['StockCode'].apply(list).to_dict()
        hits = []
        
        for (u,i) in tqdm(test_user_item_set):
            interacted_items = user_interacted_items[u]
            not_interacted_items = set(all_items_labels) - set(interacted_items)
            selected_not_interacted = list(list(np.random.choice(list(not_interacted_items), 99)))
            test_items = selected_not_interacted + [i]
            
            # Model predicted embeddings and labels
            user_embedding, item_embedding, joint_embedding, predicted_labels = model(torch.tensor([u]*100), 
                                                torch.tensor(test_items))
            
            predicted_labels = np.squeeze(predicted_labels.detach().numpy())
            
            top10_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()]
            
            if i in top10_items:
                hits.append(1)
            else:
                hits.append(0)
                
        return np.average(hits)
    
    except Exception as e:
            logger.info("Error while running model evaluation")
            logger.exception("Error in eval function " + str(e))
        
if __name__ == '__main__': 
    
    args = parse_args()
    # Load the input *.csv file as a Pandas DataFrame
    df = pd.read_csv("./app/data.csv", encoding="ISO-8859-1")
    
    # Log all the details of the input data
    shape_init = df.shape
    logger.info('#'*20 + 'SHAPE OF THE INPUT DATA' + '#'*20)
    logger.info(f'Total number of rows and columns in the initial dataset is : {shape_init[0]} and {shape_init[1]} respectively\n\n')
    logger.info('#'*20 + 'STATISTICS OF THE INPUT DATA' + '#'*20)
    # Statistical summary for numerical variables
    logger.info(f'Statistics of the numerical variables is :\n{df.describe().T}\n\n')
    # Statistical summary for categorical variables
    logger.info(f'Statistics of the categorical variables is :\n{df.describe(include="object").T}\n\n')
    
    # Cleaned df
    df = data_cleaning(df)
    
    # Map all the unique StockCode values to labels of ints for the PyTorch to process as tensors
    all_items = df['StockCode'].unique().tolist()
    items_to_labels = dict(zip(all_items, list(range(len(all_items)))))
    labels_to_items = dict(zip(list(range(len(all_items))), all_items))
    df['StockCode'] = df['StockCode'].map(items_to_labels)
    
    # Map all the unique CustomerID values to labels of ints for the PyTorch to process as tensors
    all_users = df['CustomerID'].unique().tolist()
    users_to_labels = dict(zip(all_users, list(range(len(all_users)))))
    labels_to_users = dict(zip(list(range(len(all_users))), all_users))
    df['CustomerID'] = df['CustomerID'].map(users_to_labels)
    
    # Log all the details after cleaning the data
    shape_clean = df.shape
    perc_removed = ((shape_init[0] - shape_clean[0]) / shape_init[0]) * 100
    logger.info('#'*20 + 'DATA AFTER CLEANING' + '#'*20)
    logger.info(f'Total number of rows and columns in the initial dataset is : {shape_clean[0]} and {shape_clean[1]} respectively, after dropping {perc_removed}% of the rows\n\n')
    logger.info(f'The total unique items are {df["StockCode"].nunique()} and unique customers are {df["CustomerID"].nunique()}\n\n')
    
    #################################################################################################################################################
    # Train-test split, for each CustomerID, their most recent transaction is used as the test set(i.e. leave one out) and the rest as training set.
    # This train-test split strategy is used when training and evaluating recommender systems. Doing a random split (80/20) would not be fair, 
    # introducing data leakage with a look-ahead bias, and the performance of the trained model would not be generalizable to real-world performance.
    #################################################################################################################################################
    df['rank_latest'] = df.groupby(['CustomerID'])['InvoiceDate'] \
                                .rank(method='first', ascending=False)                          
    df_train = df[df['rank_latest'] != 1]
    df_test = df[df['rank_latest'] == 1]
    
    # Dump all the required item/user --> label mapping and cleaned_df as pickle objects further used while testing/deployment
    dir_pickle = '../pickle_dump'
    pickle_dump_files = {'items_to_labels' : items_to_labels, 'labels_to_items' : labels_to_items, 'users_to_labels' : users_to_labels, 'labels_to_users' : labels_to_users,
                         'df_test_cleaned' : df_test, 'df_cleaned' : df}
    dump_pickle(dir_pickle, pickle_dump_files)

    # drop columns that are no longer required
    df_train = df_train[['StockCode', 'CustomerID']]
    df_test = df_test[['StockCode', 'CustomerID']]
    # labels=1 to capture the interaction between a CustomerID and a StockCode
    df_train.loc[:, 'labels'] = 1
    
    # Instantiate the model
    all_items_labels = df['StockCode'].unique()
    num_users = df['CustomerID'].nunique()
    num_items = df['StockCode'].nunique()
    model = NCF(num_users, num_items, df_train, all_items_labels)
    
    # Training the model
    logger.info('#'*20 + 'TRAINING THE MODEL' + '#'*20)
    logger.info('TRAINING SUCCESSFULLY STARTED\n\n')
    trainer = pl.Trainer(max_epochs=args.epoch, accelerator=args.accelerator, num_nodes=1, reload_dataloaders_every_n_epochs=1,
                     enable_progress_bar=True, logger=csv_logger)
    trainer.fit(model)
    logger.info('TRAINING SUCCESSFULLY FINISHED\n\n')
    
    # Evaluating the model performance on the test dataset
    hits_avg = eval(df, df_test, all_items_labels, model)
    logger.info('#'*20 + 'EVALUATION RESULT HR@10' + '#'*20)
    logger.info("The Hit Ratio @ 10 is {:.2f}".format(hits_avg))
    