import torch
import torchvision

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

dataset = torchvision.datasets.Omniglot(
    root="./data", download=True, transform=torchvision.transforms.ToTensor()
)
num_all_classes = 964
req_classes = 10
train_per_class = 1
test_per_class = 1

def train_test():
	
	train_samples = {}
	test_samples = {}

	for i in range(len(dataset)):
		image, label = dataset[i]
		image = image.to(device)

		if len(test_samples) > req_classes:
			break

		if label not in train_samples:
			train_samples[label] = [image]
			continue
		else:
			if len(train_samples[label]) < train_per_class:
				train_samples[label].append(image)
				continue

		if label in train_samples:
			if label not in test_samples:
				test_samples[label] = [image]
				continue
			else:
				if len(test_samples[label]) < test_per_class:
					test_samples[label].append(image)
					continue
					
	train_imgs = []
	train_labels = []
	for l, imgs in train_samples.items():
		train_imgs += imgs
		train_labels += [l] * len(imgs)

	test_imgs = []
	test_labels = []
	for l, imgs in test_samples.items():
		test_imgs += imgs
		test_labels += [l] * len(imgs)

	train_batch = torch.stack(train_imgs, dim=0)
	train_batch = train_batch.expand(train_batch.shape[0], 3, train_batch.shape[2], train_batch.shape[3])
	
	test_batch = torch.stack(test_imgs, dim=0)
	test_batch = test_batch.expand(test_batch.shape[0], 3, test_batch.shape[2], test_batch.shape[3])

	return { 'data': train_batch, 'labels': torch.tensor(train_labels) }, \
				 { 'data': test_batch, 'labels': torch.tensor(test_labels) }