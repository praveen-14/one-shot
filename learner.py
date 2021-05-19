import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from utils import progress_bar
from pytorch_pretrained_vit import ViT

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		# self.backbone = models.resnet18(pretrained=True).to(device)
		self.backbone = model = ViT('B_16_imagenet1k', pretrained=True).to(device)
		for name, param in self.backbone.named_parameters():
			param.requires_grad = False
			# print(name, param.shape)

		# self.backbone.fc = nn.Linear(512, 10).to(device)
		self.backbone.fc = nn.Linear(768, 10).to(device)

	def forward(self, x):
		return self.backbone(x)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

model = Model()
criterion = nn.CrossEntropyLoss()
optm = optim.SGD([x for x in model.parameters() if x.requires_grad], lr=1e-3, weight_decay=1e-4, momentum=0.9)

def learn(inputs: torch.Tensor, targets: torch.Tensor):
	inputs = torch.cat([inputs, torch.flip(inputs, (3,))], dim=0)
	targets = torch.cat([targets] * 2, dim=0)
	itrs = 100
	for i in range(itrs):
		model.train()
		train_loss = 0
		correct = 0
		total = 0

		optm.zero_grad()
		with torch.enable_grad():
			outputs = model(inputs)
			loss = criterion(outputs, targets)
		loss.backward()		
		optm.step()
		print(loss)

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()

		progress_bar(i, itrs, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
		  % (train_loss/(i+1), 100.*correct/total, correct, total))

def predict(input: torch.Tensor):
	_, predicted = model(input).max(1)
	return predicted


# def learn(input: torch.Tensor, target: torch.Tensor):
# 	global vec
# 	global labels
# 	vec = (embed(input) + embed(torch.flip(input, (3,))) + embed(torch.flip(input, (2,))) + embed(torch.flip(input, (2, 3)))) / 4
# 	labels = target