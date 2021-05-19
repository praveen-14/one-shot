import torch
import learner
import lfw as dataset
import preprocess
import torchvision.transforms as transforms

with torch.no_grad():
	train_set, test_set = dataset.train_test()
	learner.learn(preprocess.resize_with_padding(train_set['data'], 390, 390), train_set['labels'])
	preds = learner.predict(preprocess.resize_with_padding(test_set['data'], 390, 390))
	print("predictions\t=", preds.tolist())
	print("ground truth \t=", test_set['labels'].tolist())
	print("accuracy\t= %.2f%%" % ((preds == test_set['labels']).sum().item() * 100 / preds.numel(),))