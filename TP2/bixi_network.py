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
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 2)
        self.fc4 = nn.Linear(2, 1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = F.sigmoid(self.fc4(X))
        return X
