print(f"\n======= nn.Module========")
import torch.nn as nn

#inherit  from nn.Module
class LinerRegressionModel(nn.Module):
    def __init__(self, input_features, out_features):
        super().__init__()
        # in the constructor define the layer we will use.
        self.linear_layer = nn.Linear(input_features, out_features)
    def forward(self, x):
        # in the forward pass, we connect the layers
        return self.linear(x)
#instantiate the model 
model = LinerRegressionModel(input_features = 1, out_features= 1)
print("model architecture:")
print(model)

print(f"\n======= torch.optim ========")
import torch.optim as optim

# Hyperparameters
learning_rate = 0.01

# create an Adam Optimizer
# pass model.parameters() to tell it whic tensors to manage.PendingDeprecationWarning
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# also grab a pre-built loss function from torch.nn
loss_fn = nn.MSELoss() #mean squared loss
"""
#1 
optimizer.zero_grad()
#2 
loss.backward()
#3 
optimizer.step()
"""
