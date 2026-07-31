# forward pass "models 1st guess"
# goal: implement a models 1st guess using only the raw tensor operations 

# simple linear regression 
# (y^= xW + b)
# X is input data, W is the weight, b is the bias.set

# batch of data will have 10 data points
# eqn: y = 2x + 1
N = 10
#each data pt has 1 input feature and 1 output value
D_in = 1
D_out= 1
import torch
#create our input data X
X = torch.randn(N, D_in)

#create our ture target lables y using the "true" W and b
#the true w is 2.0 and b is 1.0
true_W = torch.tensor([[2.0]])
true_b = torch.tensor(1.0)
y_true = X @ true_W + true_b + torch.randn(N, D_out) * 0.1 # add a little noise


#initilize parameters with random values.
#shapes must be correct for matrix multiplucation

W = torch.randn(D_in, D_out, requires_grad=True)
b = torch.randn(1, requires_grad=True)

print(f"initial weight W:\n {W}\n")
print(f"initial bias b:\n {b}")

# Calculate the forward pass using manual matrix multiplication
y_hat = X @ W + b

print(f"Predictions (y_hat):\n {y_hat}") # not accurate 


# compair guess to truth 
# MSE ==> Mean Square Error (most common loss function)
"""
for every prediction, find the difference (y_hat - y)
square that difference, to make it positive (y_hat - y)^2
take the avg of all those squared differences.
"""
error = y_hat - y_true 
squared_error = error ** 2
loss = squared_error.mean()
print(f"\nloss (single scorecard number): {loss}")


# travel backward from loss and calculate gradients for all parameters with 'requires_grad = True'
# the gradint of the loss w.r.t weight W 
# the gradint of the loss w.r.t bias b 

loss.backward() # compute gradients

# the geadients are now stored in the .grad attribute
print(f"gradient for W:\n {W.grad}\n")
print(f"gradient for b:\n {b.grad}")


#traning
"""
θ_{t+1} = θ_t - η * ∇_θ L

- θ (theta) represents all our parameters. For us, that's W and b.
- η (eta) is the learning rate. A small number (e.g., 0.01) controlling our step size.
- ∇_θ L is the gradient of the loss. We just calculated this! It's in W.grad and b.grad.

W_new = W_old - learning_rate * W.grad
b_new = b_old - learning_rate * b.grad
"""
# torch.no_grad():  dont track parameter updates 
# .grad.zero_():  reset gradients each iteration  

# hyperparameters
learning_rate , epochs = 0.01 ,500 # more would be better i guess 

# Re-initialize parameters
W, b = torch.randn(1, 1, requires_grad = True), torch.randn(1, requires_grad=True)
"""
#traning loop 
for epoch in range (epochs): 
    #forward pass and loss
    y_hat = X @ W + b
    loss = torch.mean((y_hat - y_true)**2)

    #backward pass
    loss.backward()

    #update parameters 
    with torch.no_grad():
        W -= learning_rate * W.grad; b -= learning_rate * b.grad

    #zero  gradints 
    W.grad.zero_(); b.grad.zero_()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d}: loss={loss.item():.4f}, W={W.item():.3f}, b={b.item():.3f}")

print(f"\n final Parameters: W={W.item():.3f}, b={b.item():.3f}")
print(f"True Parameters: W=2.00, b=1.000")
print(f"=================================================")
"""

# torchnn or NN.linear

#the input has 2 feature, output has 1 value
D_in = 1
D_out = 1

# create the linear layer 
linear_layer = torch.nn.Linear(in_features = D_in , out_features = D_out)

# parameters created 
print(f"\nlayer's weight W: {linear_layer.weight}\n")
print(f"layer's Bias b: {linear_layer.bias}")

# assume x is a tensor of shape (10, 1) from above 
y_hat_nn = linear_layer(X)
print(f"Output of nn.Linear(first 3 rows):\n {y_hat_nn[:3]}")
print(f"======= NN.ReLu ========")
# NN.RELU to and kinks i guess 
# if an input is negative, make it zero ReLu(x) = max(0,x)
relu = torch.nn.ReLU()
sample_data = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
activated_data = relu(sample_data)

print(f"original Data: {sample_data}")
print(f"Data after ReLu: {activated_data}")

print(f"======= NN.GELU ========")
#smooth, gentily curving version of ReLU std for teansformers
gelu = torch.nn.GELU()
sample_data = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
activated_data = gelu(sample_data)

print(f"original Data: {sample_data}")
print(f"Data after GELU: {activated_data}\n")

print(f"======= NN.SOFTMAX ========")
# used on final output for classification
# convert logits==> probability distribution i.e values E [0, 1] and sum = 1
softmax = torch.nn.Softmax(dim=-1)
logits = torch.tensor([[1.0, 3.0, 0.5, 1.5], [-1.0, 2.0, 1.0, 0.0]])
probablities = softmax(logits)

print(f"output probabilities:\n {probablities}")
print(f"sum of probabilities for item 1: {probablities[0].sum()}\n")



print(f"======= NN.EMBEDDING ========")
# ==================== imp =========================
# NN.EMBEDDING   words ===> numbers
vocab_size = 10  # lang has 10  unique words
embedding_dim = 3 # represent each word with a 3d vector

embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)

#input: A sentence where each word is an ID ex: 1, 5, 0, 8
input_ids = torch.tensor([[1, 5, 0 , 8]])
word_vectors = embedding_layer(input_ids)

print({input_ids})
print({word_vectors})


print(f"======= NN.LAYERNORM ========")
# prevent values from exploding/vanishing
# above word vector have a feature dimension of 3
norm_layer = torch.nn.LayerNorm(normalized_shape=3)
input_features = torch.tensor([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])
normalized_features = norm_layer(input_features)

print(f"mean(should be ~ 0): {normalized_features}")
print(f"sdt dev (should be ~ 1): {normalized_features.std(dim= -1)}")


print(f"\n======= NN.DROPOUT ========")
# prevents overfitting
dropout_layer = torch.nn.Dropout(p=0.5)
input_tensor = torch.ones(1, 10)

#activates dropout for traning
dropout_layer.train()
output_during_train = dropout_layer(input_tensor)

#deactivate dropout for evaluvation/ prediction
dropout_layer.eval()
output_during_eval = dropout_layer(input_tensor)

print(f"output during traning: {output_during_train}")
print(f"output during eval: {output_during_eval}")



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
        return self.linear_layer(x)
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
epochs = 100
for epoch in range (epochs):
    y_hat = model(X) #forward pass
    loss = loss_fn(y_hat, y_true) # calculate loss
    optimizer.zero_grad() # zero the gradints
    loss.backward() # compute gradints
    optimizer.step()# update the parameter

    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d}  loss={loss.item():.4f}")
