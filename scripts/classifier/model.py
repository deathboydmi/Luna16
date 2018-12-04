import torch
import torch.nn as nn

class NodulesClassifier(nn.Module):
	def __init__(self, num_classes=2):
		super(NodulesClassifier, self).__init__()
		self.num_classes = num_classes
		self.features = nn.Sequential(
			nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(num_features=4),
			nn.ReLU(),
			
			nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(num_features=16),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2), # size = 64*64

			nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(num_features=32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2), # size = 32*32
			
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(num_features=64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2), # size = 16*16
			
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(num_features=128),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2), # size = 8*8
			
			nn.Conv2d(128, 256, kernel_size=2, stride=2), # size = 4*4
			nn.BatchNorm2d(num_features=256),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2) # size = 2*2
		)
		self.gaap = nn.AdaptiveAvgPool2d((1, 1))
		self.classifier = nn.Sequential(
			nn.Conv2d(256, 64, kernel_size=1),
			nn.Tanh(),
			nn.Conv2d(64, self.num_classes, kernel_size=1),
			nn.AdaptiveAvgPool2d((1, 1))
		)

		for f in self.features:
			if isinstance(f, nn.Conv2d):
				nn.init.kaiming_normal_(f.weight, nonlinearity='relu')
				nn.init.constant_(f.bias, 0)
			elif isinstance(f, nn.BatchNorm2d):
				nn.init.constant_(f.weight, 1)
				nn.init.constant_(f.bias, 0)
		for c in self.classifier:
			if isinstance(f, nn.Conv2d):
				nn.init.xavier_normal_(c.weight)
				nn.init.constant_(c.bias, 0)


	def forward(self, x):
		x = self.features(x)
		x = self.gaap(x)
		x = self.classifier(x)

		return x.view(x.size(0), self.num_classes)

if __name__ == "__main__":
	model = NodulesClassifier()
	print(model)