import torch
import torchvision
from torchvision import models, transforms
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler

# Our image transformations for the training set
transform = transforms.Compose([
    transforms.RandomResizedCrop(224,scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
# Set the batch size
batch_size = 10

# Load the training and validation datasets
data_dir = '/mnt/c/Users/theod/OneDrive/Documents/ULB/Ma0/Q2/INFO-H-410_TechniquesOfArtificialIntelligence/Projet_IA/dogs-vs-cats'
train_dataset = torchvision.datasets.ImageFolder(data_dir+'/train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=2)
val_dataset = torchvision.datasets.ImageFolder(data_dir+'/val', transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
classes = {'dog', 'cat'}

# Check if CUDA is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.is_available())

# Create the network
net = models.resnet50(pretrained=True)

# Replace the last layer to match the number of classes in the new dataset
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)  # 2 Class : cats and dogs

# Move the model to the device (GPU or CPU)
net = net.to(device)

# Define function of loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.fc.parameters(), lr=0.001, momentum=0.9)

#Scheduler for Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


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

        # Step the scheduler at the end of each epoch       #Fine tunig 
        exp_lr_scheduler.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 2000 == 1999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            running_loss = 0.

    return last_loss

# Train the network
epoch_number = 0
Name = 'Fine_tuning'
EPOCHS = 30
best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    net.train(True)
    avg_loss = train_one_epoch(epoch_number)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    net.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = net(vinputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))


    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model/model_{}_{}'.format(Name, epoch_number)
        torch.save(net.state_dict(), model_path)

    epoch_number += 1

print('Finished Training')