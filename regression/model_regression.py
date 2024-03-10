import torch
import torch.nn as nn
import torch.nn.functional as F

class regressionUnitPrice(nn.Module):
    """ regressionUnitPrice
    
        Args:
            x (torch.tensor): All the independent variables required for 'UnitPrice' prediction
    """
    def __init__(self):
        super(regressionUnitPrice, self).__init__()
        
        self.layers = nn.Sequential(
        nn.Linear(4, 1024),
        nn.LeakyReLU(),
        nn.Linear(1024, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 32),
        nn.LeakyReLU(),
        nn.Linear(32, 1)
        )

    def forward(self, x):
        # Pass through sequential layer
        pred = self.layers(x)
        return pred