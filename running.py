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
import gc

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
				epoch, batch_idx * len(data), (len(train_loader_lr)*len(data)),
				100. * batch_idx / len(train_loader_lr), loss.item()))
		
		if batch_idx % 1000 == 0:
			sample(model, data, target, len(data), mu=1.1, step=epoch*batch_idx)



def test(args, model, device, test_loader_lr, test_loader_hr):
	model.eval()
	test_loss = 0
	with torch.no_grad():
		for (data, t), (target, p) in zip(test_loader_lr, test_loader_hr):
			data, target = data.to(device), target.to(device)
			prior_logits, conditioning_logits = model(lr_images = data, hr_images = target)
			l1 = softmax_loss(prior_logits+conditioning_logits, target)
			l2 = softmax_loss(conditioning_logits, target)
			test_loss += l1+l2 # sum up batch loss

	test_loss /= len(test_loader_lr)*32
	print("test_loss : ", test_loss.item())

def logits_2_pixel_value(logits, mu=1.1):
	# print("convert")
	rebalance_logits = logits * mu
	probs = softmax(rebalance_logits)
	pixel_dict = torch.arange(0, 256, dtype=torch.float32).to("cuda")
	pixels = torch.sum(probs*pixel_dict, dim=1)
	return pixels/255

def softmax(x):
	# print("cal")
	"""Compute softmax values for each sets of scores in x."""
	# x = x.detach().numpy()
	a, b = torch.max(x, -1, keepdim=True, out=None)
	e_x = torch.exp(x - a)
	return e_x / e_x.sum(dim=-1, keepdim =True) # only difference

def sample(model, data, target, batch_size, mu=1.1, step=None):
	with torch.no_grad():
		np_lr_imgs = data
		np_hr_imgs = target
		c_logits = model.conditioning_network
		p_logits = model.prior_network
		gen_hr_imgs = torch.zeros((batch_size, 3, 32, 32), dtype=torch.float32).to("cuda")
		np_c_logits = c_logits(np_lr_imgs)
		for i in range(32):
			for j in range(32):
				for c in range(3):
					print(i,j,c)
					np_p_logits = p_logits(gen_hr_imgs)
					new_pixel = logits_2_pixel_value(np_c_logits[:, c*256:(c+1)*256, i, j] + np_p_logits[:, c*256:(c+1)*256, i, j], mu=mu)
					gen_hr_imgs[:, c, i, j] = new_pixel
					# try:
					# 	for obj in gc.get_objects():
					# 		if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
					# 			print(type(obj), obj.size())
					# except:
					# 	print("error")
		samples_dir = "/home/eee/ug/15084005/DIH/samples/"
		print("sample")
		save_samples(np_lr_imgs, samples_dir + '/lr_' + str(mu*10) + '_' + str(step))
		save_samples(np_hr_imgs, samples_dir + '/hr_' + str(mu*10) + '_' + str(step))
		save_samples(gen_hr_imgs, samples_dir + '/generate_' + str(mu*10) + '_' + str(step))

def save_samples(np_imgs, img_path):
	print("save")
	torchvision.utils.save_image(np_imgs[0, :, :, :], img_path+".jpg")

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

	data = datasets.ImageFolder(root = '/home/eee/ug/15084005/DIH/CelebA/CelebA/train/', transform = transforms.Compose([transforms.Resize((8,8), interpolation=2), transforms.ToTensor()]))
	target = datasets.ImageFolder(root = '/home/eee/ug/15084005/DIH/CelebA/CelebA/train/', transform = transforms.Compose([transforms.Resize((32,32), interpolation=2), transforms.ToTensor()]))

	data_test = datasets.ImageFolder(root = '/home/eee/ug/15084005/DIH/CelebA/CelebA/test/', transform = transforms.Compose([transforms.Resize((8,8), interpolation=2), transforms.ToTensor()]))
	target_test = datasets.ImageFolder(root = '/home/eee/ug/15084005/DIH/CelebA/CelebA/test/', transform = transforms.Compose([transforms.Resize((32,32), interpolation=2), transforms.ToTensor()]))
	

	#test_split = .05
	#shuffle_dataset = True
	#random_seed= 42
	# Creating data indices for training and validation splits:
	# dataset_size = len(image_data_hr)
	# indices = list(range(dataset_size))
	# split = int(np.floor(test_split * dataset_size))
	# if shuffle_dataset :
	# 	np.random.seed(random_seed)
	# 	np.random.shuffle(indices)
	# train_indices, test_indices = indices[split:], indices[:split]

	# data = [image_data_lr[x] for x in train_indices]
	# target = [image_data_hr[x] for x in train_indices]

	# data_test = [image_data_lr[x] for x in train_indices]
	# target_test = [image_data_hr[x] for x in train_indices]

	train_sampler_lr = torch.utils.data.sampler.SequentialSampler(data)
	train_sampler_hr = torch.utils.data.sampler.SequentialSampler(target)

	test_sampler_lr = torch.utils.data.sampler.SequentialSampler(data_test)
	test_sampler_hr = torch.utils.data.sampler.SequentialSampler(target_test)

	train_loader_lr = torch.utils.data.DataLoader(data, batch_size=args.batch_size, sampler=train_sampler_lr)
	train_loader_hr = torch.utils.data.DataLoader(target, batch_size=args.batch_size, sampler=train_sampler_hr)

	test_loader_lr = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, sampler=test_sampler_lr)
	test_loader_hr = torch.utils.data.DataLoader(target_test, batch_size=args.batch_size, sampler=test_sampler_hr)

	model = Net()
	model = model.to(device)

	optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.95)
	for epoch in range(1, args.num_epoch + 1):
		train(args, model, device, train_loader_lr, train_loader_hr, optimizer, epoch)
		# sample(model, train_loader_lr, train_loader_hr, mu=1.1, step=epoch)
		test(args, model, device, test_loader_lr, test_loader_hr)

if __name__ == '__main__':
	main()
