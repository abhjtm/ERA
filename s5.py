import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


from utils.util_s5 import train_transforms, test_transforms, show_sample_images, create_training_plots
from models.model_s5 import Net, TrainTest
# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)
device = torch.device("cuda" if cuda else "cpu")

train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

batch_size = 512

kwargs = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 2, 'pin_memory': True}

test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
train_loader = torch.utils.data.DataLoader(train_data, **kwargs)


show_sample_images(train_loader)


model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
# New Line
criterion = nn.CrossEntropyLoss()
num_epochs = 20

tt = TrainTest()
for epoch in range(1, num_epochs+1):
  print(f'Epoch {epoch}')
  tt.train(model, device, train_loader, optimizer, criterion)
  tt.test(model, device, train_loader, criterion)
  scheduler.step()


create_training_plots(tt.train_losses, tt.train_acc, tt.test_losses, tt.test_acc)