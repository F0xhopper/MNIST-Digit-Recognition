# Import necessary libraries
from torchvision import datasets  # For loading datasets
from torchvision.transforms import ToTensor  # For transforming data to PyTorch tensors
from torch.utils.data import DataLoader  # For handling data loading in batches
import torch.nn as nn  # PyTorch's neural network module
import torch.nn.functional as F  # PyTorch's functional interface
import torch.optim as optim  # For optimization algorithms
import torch  # PyTorch library
import ssl  # For SSL certificate handling
import matplotlib.pyplot as plt  # For plotting images

# Ignore SSL certificate errors
ssl._create_default_https_context = ssl._create_unverified_context

# Download and load the MNIST dataset for training
training_data = datasets.MNIST(
    root='data',  # Root directory where the dataset is located or where it will be saved
    train=True,  # Indicates that this is the training set
    transform=ToTensor(),  # Transform PIL Image or numpy.ndarray to tensor
    download=True  # Downloads the dataset if not found in root directory
)

# Download and load the MNIST dataset for testing
testing_data = datasets.MNIST(
    root='data',  # Root directory where the dataset is located or where it will be saved
    train=False,  # Indicates that this is the testing set
    transform=ToTensor(),  # Transform PIL Image or numpy.ndarray to tensor
    download=True  # Downloads the dataset if not found in root directory
)

# Create data loaders for training and testing data
loaders = {
    'train': DataLoader(training_data, batch_size=100, shuffle=True, num_workers=1),  # DataLoader for training data
    'test': DataLoader(testing_data, batch_size=100, shuffle=True, num_workers=1),  # DataLoader for testing data
}

# Define the Convolutional Neural Network (CNN) model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # Input channels: 1, Output channels: 10, Kernel size: 5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # Input channels: 10, Output channels: 20, Kernel size: 5x5
        self.conv2_drop = nn.Dropout2d()  # Dropout layer to prevent overfitting
        # Define fully connected layers
        self.fc1 = nn.Linear(320, 50)  # Input size: 320, Output size: 50
        self.fc2 = nn.Linear(50, 10)  # Input size: 50, Output size: 10

    def forward(self, x):
        # Perform forward pass through the network
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # Apply max pooling and ReLU activation
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # Apply convolution, dropout, max pooling, and ReLU activation
        x = x.view(-1, 320)  # Reshape the tensor
        x = F.relu(self.fc1(x))  # Apply ReLU activation to the first fully connected layer
        x = F.dropout(x, training=self.training)  # Apply dropout to prevent overfitting
        x = self.fc2(x)  # Output layer without activation function
        return F.softmax(x, dim=1)  # Apply softmax activation for classification

# Check for GPU availability and move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

# Define optimizer and loss function
optimiser = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001
loss_fn = nn.CrossEntropyLoss()  # Cross-entropy loss function for classification

# Define the training function
def train(epoch):
    model.train()  # Set the model to training mode
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)  # Move data to GPU if available
        optimiser.zero_grad()  # Clear the gradients of all optimized tensors
        output = model(data)  # Forward pass: compute predicted outputs by passing inputs to the model
        loss = loss_fn(output, target)  # Compute the loss
        loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
        optimiser.step()  # Perform a single optimization step (parameter update)
        if batch_idx % 20 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)} / {len(loaders['train'].dataset)} ({100 * batch_idx / len(loaders['train']):.0f}%)]\t{loss.item():.6f}")

# Define the testing function
def test():
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)  # Move data to GPU if available
            output = model(data)  # Forward pass: compute predicted outputs by passing inputs to the model
            test_loss += loss_fn(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions

    test_loss /= len(loaders['test'].dataset)  # Calculate average test loss
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders['test'].dataset)} ({100. * correct / len(loaders['test'].dataset):.0f}%)\n")

# Define a function to test a single image and display the result
def test_and_display_images_grid(num_rows=10, num_cols=10):
    model.eval()  # Set the model to evaluation mode
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            data, target = testing_data[index]  # Get data and target label from testing dataset
            data = data.unsqueeze(0).to(device)  # Add a batch dimension and move data to GPU if available
            output = model(data)  # Forward pass: compute predicted outputs by passing inputs to the model
            prediction = output.argmax(dim=1, keepdim=True).item()  # Get the predicted class label
            image = data.squeeze(0).squeeze(0).cpu().numpy()  # Move image data back to CPU and remove batch dimension
            # Display the image
            axes[i, j].imshow(image, cmap='gray')
            axes[i, j].set_title(f"Pred: {prediction}, True: {target}")
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()

# Main function to train the model and evaluate its performance
if __name__ == '__main__':
    for epoch in range(1, 10):
        train(epoch)
        test()
    test_and_display_images_grid()