from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, 1, stride=1, padding=0)
		self.Resconv = nn.Conv2d(32, 32, 3, stride=1, padding=1)
		self.Transconv = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding = 1)
		self.conv2 = nn.Conv2d(32, 3*256, 1, stride=1, padding=0)

		self.Maskconv1 = nn.Conv2d(3, 64, 7, stride=1, padding=3)
		self.Gateconv1 = nn.Conv2d(64, 2*64, 5, stride=1, padding=2)
		self.Gateconv2 = nn.Conv2d(2*64, 2*64, (1,1), stride=1, padding=0)
		self.Gateconv3 = nn.Conv2d(64, 2*64, (1,5), stride=1, padding=(0,2))
		self.Gateconv4 = nn.Conv2d(64, 64, (1,1), stride=1, padding=0)
		self.Maskconv2 = nn.Conv2d(64, 1024, 1, stride=1, padding=0)
		self.Maskconv3 = nn.Conv2d(1024, 3*256, 1, stride=1, padding=0)

		self.conv1.weight = nn.Parameter(self.weight_init(3, 32, (1,1), mask_type = None))
		self.Resconv.weight = nn.Parameter(self.weight_init(32, 32, (3,3), mask_type = None))
		self.Transconv.weight = nn.Parameter(self.weight_init(32, 32, (3,3), mask_type = None))
		self.conv2.weight = nn.Parameter(self.weight_init(32, 3*256, (1,1), mask_type = None))

		self.Maskconv1.weight = nn.Parameter(self.weight_init(3, 64, (7,7), mask_type='A'))
		self.Gateconv1.weight = nn.Parameter(self.weight_init(64, 2*64, (5,5), mask_type='C'))
		self.Gateconv2.weight = nn.Parameter(self.weight_init(2*64, 2*64, (1,1), mask_type=None))
		self.Gateconv3.weight = nn.Parameter(self.weight_init(64, 2*64, (1,5), mask_type='B'))
		self.Gateconv4.weight = nn.Parameter(self.weight_init(64, 64, (1,1), mask_type='B'))
		self.Maskconv2.weight = nn.Parameter(self.weight_init(64, 1024, (1,1), mask_type='B'))
		self.Maskconv3.weight = nn.Parameter(self.weight_init(1024, 3*256, (1,1), mask_type='B'))

	def weight_init(self, in_channel, num_outputs, kernel_shape, mask_type=None):
		kernel_h, kernel_w = kernel_shape
		center_h = kernel_h // 2
		center_w = kernel_w // 2

		mask = np.zeros((num_outputs, in_channel, kernel_h, kernel_w), dtype=np.float32)
		if mask_type is not None:
			mask[:, :, :center_h, :] = 1
			if mask_type == 'A':
				mask[:, :, center_h, :center_w] = 1
			if mask_type == 'B':
				mask[:, :, center_h, :center_w+1] = 1
		else:
			mask[:, :, :, :] = 1

		weights_shape = [num_outputs, in_channel, kernel_h, kernel_w]
		weights = torch.empty(weights_shape)
		weights = nn.init.xavier_normal_(weights)
		weights = weights*torch.from_numpy(mask)
		return weights

	def prior_network(self, hr_images):
		conv1 = self.Maskconv1(hr_images)
		inputs = conv1
		state = conv1
		for i in range(20):
			inputs, state = self.gated_conv2d(inputs, state, [5, 5])
		conv2 = self.Maskconv2(inputs)
		conv2 = F.relu(conv2)
		prior_logits = self.Maskconv3(conv2)
		prior_logits = torch.cat((prior_logits[:, 0::3, :, :], prior_logits[:, 1::3, :, :], prior_logits[:, 2::3, :, :]), 1)
		return prior_logits

	def conditioning_network(self, lr_images):
		res_num = 6
		inputs = lr_images
		inputs = self.conv1(inputs)
		for i in range(2):
			for j in range(res_num):
				inputs = self.resnet_block(inputs)
			inputs = self.Transconv(inputs)
			inputs = F.relu(inputs)
		for i in range(res_num):
			inputs = self.resnet_block(inputs)
		conditioning_logits = self.conv2(inputs)
		return conditioning_logits
	
	def batch_norm(self, x):
		bn = nn.BatchNorm2d(x.shape[1], affine=False, track_running_stats=False)
		return bn(x)

	def resnet_block(self, inputs):
		conv1 = self.Resconv(inputs)
		bn1 = self.batch_norm(conv1)
		relu1 = F.relu(bn1)
		conv2 = self.Resconv(relu1)
		bn2 = self.batch_norm(conv2)
		output = inputs + bn2
		return output


	def gated_conv2d(self, inputs, state, kernel_shape):
		batch_size, in_channel, height, width  = list(inputs.size())
		kernel_h, kernel_w = kernel_shape
		left = self.Gateconv1(state)
		left1 = left[:, 0:in_channel, :, :]
		left2 = left[:, in_channel:, :, :]
		left1 = F.tanh(left1)
		left2 = F.sigmoid(left2)
		new_state = left1 * left2
		left2right = self.Gateconv2(left)
		right = self.Gateconv3(inputs)
		right = right + left2right
		right1 = right[:, 0:in_channel, :, :]
		right2 = right[:, in_channel:, :, :]
		right1 = F.tanh(right1)
		right2 = F.sigmoid(right2)
		up_right = right1 * right2
		up_right = self.Gateconv4(up_right)
		outputs = inputs + up_right
		return outputs, new_state

	def forward(self, lr_images, hr_images):
		labels = hr_images
		hr_images = hr_images - 0.5
		lr_images = lr_images - 0.5
		prior_logits = self.prior_network(hr_images)
		conditioning_logits = self.conditioning_network(lr_images)
		return prior_logits, conditioning_logits

def softmax_loss(logits, labels):
	logits = torch.reshape(logits, [-1, 256])
	labels = labels.to(torch.int64)
	labels = torch.reshape(labels, [-1])
	return F.cross_entropy(logits, labels)

def train(args, model, device, train_loader_lr, train_loader_hr, optimizer, epoch):
	model.train()
	print(device)
	for (batch_idx, (data, t)), (target, p) in zip(enumerate(train_loader_lr), train_loader_hr):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		prior_logits, conditioning_logits = model(lr_images = data, hr_images = target)
		l1 = softmax_loss(prior_logits+conditioning_logits, target)
		l2 = softmax_loss(conditioning_logits, target)
		loss = l1+l2
		loss3 = softmax_loss(prior_logits, target)
		loss.backward()
		optimizer.step()
		if batch_idx % 5 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader_lr.dataset),
				100. * batch_idx / len(train_loader_lr), loss.item()))


def test(args, model, device, test_loader_lr, test_loader_hr):
	model.eval()
	test_loss = 0
	with torch.no_grad():
		for (data, t), (target, p) in zip(test_loader_lr, test_loader_hr):
			data, target = data.to(device), target.to(device)
			prior_logits, conditioning_logits = model(lr_images = data, hr_images = target)
			l1 = softmax_loss(prior_logits+conditioning_logits, target)
			l2 = softmax_loss(conditioning_logits, target)
			test_loss += l1+l2  # sum up batch loss
			

	test_loss /= len(test_loader_lr.dataset)
	print("test_loss : ", test_loss.item())

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--use_gpu", type = bool, default = True, help = "use or not gpu")
	parser.add_argument("--num_epoch", type = int, default = 30, help = "no of epoch")
	parser.add_argument("--batch_size", type = int, default = 32, help = "batch size")
	parser.add_argument("--learning_rate", type = float, default = 4e-4, help = "learning rate")
	args = parser.parse_args()

	use_cuda = args.use_gpu and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	torch.cuda.set_device(0)

	image_data_hr = datasets.ImageFolder(root = '/home/eee/ug/15084005/DIH/CelebA/Img/', transform = transforms.Compose([transforms.Resize((32,32), interpolation=2), transforms.ToTensor()]))
	image_data_lr = datasets.ImageFolder(root = '/home/eee/ug/15084005/DIH/CelebA/Img/', transform = transforms.Compose([transforms.Resize((8,8), interpolation=2), transforms.ToTensor()]))

	test_split = .05
	shuffle_dataset = True
	random_seed= 42
	# Creating data indices for training and validation splits:
	dataset_size = len(image_data_hr)
	indices = list(range(dataset_size))
	split = int(np.floor(test_split * dataset_size))
	if shuffle_dataset :
		np.random.seed(random_seed)
		np.random.shuffle(indices)
	train_indices, test_indices = indices[split:], indices[:split]

	train_sampler_hr = SubsetRandomSampler(train_indices)
	# train_sampler_lr = SubsetRandomSampler(train_indices)

	test_sampler_hr = SubsetRandomSampler(test_indices)
	# test_indices_lr = SubsetRandomSampler(test_indices)

	train_loader_hr = torch.utils.data.DataLoader(image_data_hr, batch_size=args.batch_size, sampler=train_sampler_hr)
	train_loader_lr = torch.utils.data.DataLoader(image_data_lr, batch_size=args.batch_size, sampler=train_sampler_hr)

	test_loader_hr = torch.utils.data.DataLoader(image_data_hr, batch_size=args.batch_size, sampler=test_sampler_hr)
	test_loader_lr = torch.utils.data.DataLoader(image_data_lr, batch_size=args.batch_size, sampler=test_sampler_hr)

	model = Net()
	model = model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
	for epoch in range(1, args.num_epoch + 1):
		train(args, model, device, train_loader_lr, train_loader_hr, optimizer, epoch)
		test(args, model, device, test_loader_lr, test_loader_hr)

if __name__ == '__main__':
	main()
