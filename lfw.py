import torch
import torchvision
from PIL import Image
import os
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

data_path = "./data/lfw"

def train_test():
	
	label_names = []
	train_samples = []
	train_labels = []
	test_samples = []
	test_labels = []

	p = os.path.join(data_path, "train")
	for subdir, dirs, files in os.walk(p):
		if subdir != p:
			class_name = os.path.basename(subdir)
			label_names.append(class_name)

	def read_data(data, target, train_val):
		p = os.path.join(data_path, train_val)
		for subdir, dirs, files in os.walk(p):
			if subdir != p:
				class_name = os.path.basename(subdir)
				with Image.open(os.path.join(subdir, files[0])) as im:
					data.append(transforms.ToTensor()(im))
				target.append(label_names.index(class_name))


	read_data(train_samples, train_labels, "train")
	read_data(test_samples, test_labels, "test")

	train_samples = torch.stack(train_samples, dim=0).to(device)
	train_labels = torch.tensor(train_labels).to(device)
	test_samples = torch.stack(test_samples, dim=0).to(device)
	test_labels = torch.tensor(test_labels).to(device)

	return { 'data': train_samples, 'labels': train_labels }, \
				 { 'data': test_samples, 'labels': test_labels }
