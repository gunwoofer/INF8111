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
        # self.fc1 = nn.Linear(222, 140)
        # self.fc2 = nn.Linear(140, 85)
        # self.fc3 = nn.Linear(85, 50)
        # self.fc4 = nn.Linear(50, 10)
        # self.fc5 = nn.Linear(10, 1)
        self.fc1 = nn.Linear(40, 20)
        self.fc2 = nn.Linear(20, 1)
        # self.fc3 = nn.Linear(17, 10)
        # self.fc4 = nn.Linear(10, 6)
        # self.fc5 = nn.Linear(6, 4)
        # self.fc6 = nn.Linear(4, 1)






    def forward(self, X):
        X = F.relu(self.fc1(X))
        # X = F.relu(self.fc2(X))
        # X = F.relu(self.fc3(X))
        # X = F.relu(self.fc4(X))
        # X = F.relu(self.fc5(X))
        X = F.sigmoid(self.fc2(X))
        return X
