import torch
import torch.nn as nn
import torch.optim as optim

inputs = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

expected_outputs = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [1.0]
])


class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(2, 1)
    
    def forward(self, x):
        transformation = self.linear(x)
        # output = torch.heaviside(transformation, torch.tensor(0.0)) # can't differentiate!!
        output = torch.sigmoid(transformation)
        return output


model = Perceptron()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.3)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(inputs)
    loss = criterion(outputs, expected_outputs)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# Evaluate the model
with torch.no_grad():
    outputs = model(inputs)
    print("Input\tExpected\tOutput")
    for i in range(len(inputs)):
        print(f"{inputs[i].tolist()}\t{expected_outputs[i].item()}\t{outputs[i].item()}")
