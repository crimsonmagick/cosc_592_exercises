import torch
import torch.nn as nn
import torch.optim as optim

# Define the inputs and labels
inputs = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

labels = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [1.0]
])


# Define the Perceptron model with the Heaviside step function
class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(2, 1)
    
    def forward(self, x):
        transformation = self.linear(x)
        output = torch.heaviside(transformation, torch.tensor(0.0))
        return output


# Initialize the model, loss function, and optimizer
model = Perceptron()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    outputs = model(inputs)
    print("Input\tExpected\tOutput")
    for input, label, output in zip(inputs, labels, outputs):
        print(f"{input.tolist()}\t{label.item()}\t{output.item()}")
