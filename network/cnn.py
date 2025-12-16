import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)

class LeNet(nn.Module):
	def __init__(self):
		super(LeNet, self).__init__()
		self.conv0 = nn.Conv2d(1, 6, kernel_size=5)
		self.conv1 = nn.Conv2d(6, 16, kernel_size=5)
		self.fc0 = nn.Linear(16*4*4, 120)
		self.fc1 = nn.Linear(120, 84)
		self.fc2 = nn.Linear(84, 10)
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
	
	def forward(self, x):
		# 28*28*1 -> 24*24*6 -> 12*12*6
		x = self.relu(self.conv0(x))
		x = self.maxpool(x)
		# 12*12*6 -> 8*8*16 -> 4*4*16
		x = self.relu(self.conv1(x))
		x = self.maxpool(x)

		# MLP
		x = x.view(x.size(0), -1) # 批量大小 * 特征数
		x = self.relu(self.fc0(x))
		x = self.relu(self.fc1(x))
		x = self.fc2(x)
		return x

# Data loading and preprocessing
def load_data(batch_size=64):
			transform = transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize((0.1307,), (0.3081,))
			])
			
			train_dataset = torchvision.datasets.MNIST(
					root='./data', 
					train=True,
					download=True, 
					transform=transform
			)
			
			test_dataset = torchvision.datasets.MNIST(
					root='./data', 
					train=False,
					download=True, 
					transform=transform
			)
			
			train_loader = DataLoader(
					train_dataset, 
					batch_size=batch_size,
					shuffle=True
			)
			
			test_loader = DataLoader(
					test_dataset, 
					batch_size=batch_size,
					shuffle=False
			)
			
			return train_loader, test_loader

	# training
def train(model, train_loader, criterion, optimizer, device):
			model.train()
			running_loss = 0.0
			correct = 0
			total = 0

			for images, labels in train_loader:
					images, labels = images.to(device), labels.to(device)

					optimizer.zero_grad()
					outputs = model(images)
					loss = criterion(outputs, labels)
					loss.backward()
					optimizer.step()

					running_loss += loss.item()
					_,predicted = outputs.max(1)
					total += labels.size(0)
					correct += predicted.eq(labels).sum().item()
			
			accuracy = 100. * correct / total
			return running_loss / len(train_loader), accuracy

def evaluate(model, test_loader, criterion, device):
			model.eval()
			running_loss = 0.0
			correct = 0
			total = 0
			
			with torch.no_grad():
					for images, labels in test_loader:
							images, labels = images.to(device), labels.to(device)
							outputs = model(images)
							loss = criterion(outputs, labels)
							
							running_loss += loss.item()
							_, predicted = outputs.max(1)
							total += labels.size(0)
							correct += predicted.eq(labels).sum().item()
			
			accuracy = 100. * correct / total
			return running_loss / len(test_loader), accuracy

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    
    # Load data
    train_loader, test_loader = load_data(batch_size)
    
    # Initialize model, loss function, and optimizer
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print('Starting training...')
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print('-' * 50)
    
    # Save the trained model
    torch.save(model.state_dict(), 'lenet_mnist.pth')
    print('Training completed and model saved!')

if __name__ == '__main__':
    main()