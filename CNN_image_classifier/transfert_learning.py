import torch
import torchvision
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim

# Define the transformation for the transfert learning
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the resnet50 model pretrained on ImageNet
net = models.resnet50(pretrained=True)

# Freezing all the parameters of the model
for param in net.parameters():
    param.requires_grad = False

# Replace the last layer to match the number of classes in the new dataset
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)  # 2 classes: dogs and cats

# Move the model to the device (GPU or CPU)
net = net.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.fc.parameters(), lr=0.001, momentum=0.9)

#Ajout du scheduler ??#

batch_size = 10
data_dir = '/mnt/c/Users/theod/OneDrive/Documents/ULB/Ma0/Q2/INFO-H-410_TechniquesOfArtificialIntelligence/Projet_IA/dogs-vs-cats'

# Loas the training dataset
train_dataset = torchvision.datasets.ImageFolder(data_dir+'/train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=2)

# Function to train the network for one epoch
def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader, 0):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = net(inputs)

        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 2000 == 1999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            running_loss = 0.

    return last_loss

for epoch in range(10):  # loop over the dataset multiple times

    print('Epoch {}'.format(epoch))
    train_one_epoch(epoch)

print('Finished Training')
print('Do the evaluation')

# Load the test dataset
test_dataset = torchvision.datasets.ImageFolder(data_dir+'/test', transform=transform)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=2)

net.eval()  # Set the model to evaluation mode

correct = 0
total = 0

# Evaluate the model on the test dataset
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Précision du modèle sur les données de test: {}%'.format(100 * correct / total))