import torch
import torchvision
import torchvision.models as models

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

model = models.resnet18(pretrained=True).to(device)
vec = None
labels = None

def embed(batch):
	return model(batch)

def learn(input: torch.Tensor, target: torch.Tensor):
	global vec
	global labels
	vec = embed(input)
	labels = target

def predict(input: torch.Tensor):
	global vec
	global labels
	e_arr = embed(input)
	dsts = (e_arr.unsqueeze(0) - vec.unsqueeze(1)).pow(2).sum(-1).sqrt()
	return labels[dsts.argmin(0)]
