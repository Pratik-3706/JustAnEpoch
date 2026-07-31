""" source: https://youtu.be/r1bquDz5GGA?"""

from numpy import indices
import torch
from networkx.algorithms.approximation import traveling_salesman
import torch

# input: a std py list
my_data = [[1,2,3], [4,5,6]]
my_tensor = torch.tensor(my_data)
print(f"{my_tensor}")
print("================ creation form desired shape ==========================")

# creation from desired shape 
shape = (4,5) #shape tuple (2 rows, 3 columns)
one = torch.ones(shape) #tensor of 1's
zeros = torch.zeros(shape) #tensor of 0's
random = torch.randn(shape) #tensor of 0's

print(f"random tesnor: \n {random}")
print("================ mimikck ==========================")

# creation by mimicking another tensor =======================
template = torch.tensor([[1,2],[3,4]]) #template            ||
rand_like = torch.rand_like(template, dtype = torch.float)  #<= can change data type

print(f"template_tensor:\n {template}\n")
print(f"rand_like tensor:\n {rand_like}")

print("================ 3 attributes ==========================")


#3 critical attributes:
tensor = torch.randn(2,3)
print(f"shape: {tensor.shape}")
print(f"shape: {tensor.dtype}")
print(f"shape: {tensor.device}")

print("================ autograd ==========================")
# auto grad (automatic differentiation)
# it will automatically calculate the gradient of the tensor
# off by default, to tell pythorch its a learnable parameter
requires_grad = True
# a std tensor
x_data = torch.tensor([[1.,2.],[3.,4.,]])
# a parameter tensor (we need gradints)
w = torch.tensor([[1.0],[2.0]], requires_grad = True)
print(f"data tensor requires_rad: {x_data.requires_grad}")
print(f"parameter tensor requires_rad: {w.requires_grad}")

print("================ building the graph ==========================")
# z = x * y, where y = a + b
a=torch.tensor(2.0, requires_grad=True)
b=torch.tensor(3.0, requires_grad=True)
x=torch.tensor(4.0, requires_grad=True)

y = a+b # operation 1
z = x * y # operation 2
print(f"result: {z}")
print(f"grad_fn for z: {z.grad_fn}")
print(f"grad_fn for y: {y.grad_fn}")
print(f"grad_fn for a: {a.grad_fn}\n")
print("================ verbs * ve @ ==========================")

# verbs
# ('*') element wise multiplication
#rule: tensor most have exact same shape 
a = torch.tensor([[1,2],[3,4]])
b = torch.tensor([[10, 20], [30, 40]])

#this calculates:[[1*10, 2*20], [3*30, 4*40] 
element_wise_product = a * b
print(f"element wise product {element_wise_product}")

# ('@') Matrix multiplication
#rule m1 column = m2 rows 
# shape (2,3) 
m1 = torch.tensor([[1,2,3],[4, 5, 6]])
# shape (3,2) 
m2 = torch.tensor([[7,8],[9, 10], [11, 12]])

#resulting shape will be: (2 x 2)

matrix_prod = m1 @ m2
print(f"matrix product {matrix_prod}") 
# will always ise the @ for linear layer, y = XW + b

print("================ reduction/ dim ==========================")
#default dehavior: collapse the entire tensor

scores = torch.tensor([[10., 20., 30., ], [5., 10., 15.,]])

# this calculates : (10+20+30+5=15) / 6 ===>90 / 6
average_score = scores.mean()
print (f"{average_score}")

#dim  argument (lets control which direction to collapse)
""" 
scores tensor: 2 students, 3 assignments
a simple rule for 2d tensors: 
dim 0 ==> collapses the rows. operates "vertically" ↓
dim 1 ==> collapses the colimns. operates "horizontally" ↑ 
"""
scores = torch.tensor([[10., 20., 30., ], [5., 10., 15.,]])
# to get the avg for each assignments, we collapse the student dimension (dim = 0)
avg_per_assignment = scores.mean(dim = 0)
# to get the avg for each student, we collapse the assignment dimension (dim = 1)
avg_per_student = scores.mean(dim = 1)

print(f"avg per assignment: {avg_per_assignment}")
print(f"avg per student: {avg_per_student}")

"""
VISUALIZING THE COLLAPSE

| `scores`      | Assignment 1 | Assignment 2 | Assignment 3 | `mean(dim=1)` |
|---------------|--------------|--------------|--------------|---------------|
| Student 1     | 10           | 20           | 30           | 20            |
| Student 2     | 5            | 10           | 15           | 10            |
| `mean(dim=0)` | 7.5          | 15           | 22.5         |               |
"""


print("================ basoc indexing ==========================")

x = torch.arange(12).reshape(3,4)
"""
    tensor([[ 0,  1,  2,   3],
            [ 4,  5,  6,   7],      
            [ 8,  9,  10, 11]]])
"""
# get the 3rd column (at index 2)
col_2 = x[:,2]
print(f"{col_2}")

# ARGMAX  find index of the higgest value

scores = torch.tensor(
    # the best score is at index 3
    [[10, 0, 5, 20, 1], 
    # the best score is at index 1
    [1, 30, 3, 5, 0]

])

#find the index of the best score for each
best_indices = torch.argmax(scores, dim=1)
print(f"best score for each: {best_indices}")

# torch.gather() : if we need something specific 
# ex: from row 0, get emement at column 2
# from row 1, get the element at column 0
# from row 2, get the element at column 3

data = torch.tensor([
    [10, 11, 12, 13], # r 1 
    [20, 21, 22, 23], # r 2
    [30, 31, 32, 33] # r 3
])

# list of which column to get from each row
indices_to_select = torch.tensor([[2],[0],[3]])

#grather from data along dim = 1 (cloumns)
selected_val = torch.gather(data, dim =1, index =indices_to_select)
print(f"{selected_val}")