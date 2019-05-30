import torch
import torch.nn as nn
import torch.nn.functional as F

# Input :
# - mois
# - heure
# - temperature
# - weather
# - vacances
# - station 

# Output :
# One hot vector [Low - High]

class BixiNetwork(nn.Module):
    def __init__(self):
        super(BixiNetwork, self).__init__()
        self.fc1 = nn.Linear(14, 14)
        self.fc5 = nn.Linear(14, 2)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.softmax(self.fc5(X))
        return X
