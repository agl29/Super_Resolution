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
		for j in range(3):
			for i in range(6):
				exec("self.Resconv1"+str(j)+str(i)+"="+"nn.Conv2d(32, 32, 3, stride=1, padding=1)")
				exec("self.Batch_norm1"+str(j)+str(i)+"="+"nn.BatchNorm2d(32, track_running_stats=False)")
				exec("self.Resconv2"+str(j)+str(i)+"="+"nn.Conv2d(32, 32, 3, stride=1, padding=1)")
				exec("self.Batch_norm2"+str(j)+str(i)+"="+"nn.BatchNorm2d(32, track_running_stats=False)")
		self.Transconv1 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding = 1)
		self.Transconv2 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding = 1)
		self.conv2 = nn.Conv2d(32, 3*256, 1, stride=1, padding=0)

	def conditioning_network(self, lr_images):
		res_num = 6
		inputs = lr_images
		inputs = self.conv1(inputs)
		for i in range(2):
			for j in range(res_num):
				inputs = self.resnet_block(inputs, i, j)
			inputs = eval("self.Transconv"+str(i+1))(inputs)
			inputs = F.relu(inputs)
		for i in range(res_num):
			inputs = self.resnet_block(inputs, 2, i)
		conditioning_logits = self.conv2(inputs)
		return conditioning_logits

	def resnet_block(self, inputs, i, j):
		conv1 = eval("self.Resconv1"+str(i)+str(j))(inputs)
		bn1 = eval("self.Batch_norm1"+str(i)+str(j))(conv1)
		relu1 = F.relu(bn1)
		conv2 = eval("self.Resconv2"+str(i)+str(j))(relu1)
		bn2 = eval("self.Batch_norm2"+str(i)+str(j))(conv2)
		output = inputs + bn2
		return output


	def forward(self, lr_images):
		lr_images = lr_images - 0.5
		conditioning_logits = self.conditioning_network(lr_images)
		return conditioning_logits


####  data load
from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data import DataLoader
import torchvision


def input_transform(crop_size):
	return Compose([Resize((8,8),interpolation = 2),ToTensor(),])
def target_transform(crop_size):
	return Compose([Resize((32,32),interpolation = 2),ToTensor(),])
def is_image_file(filename):
	return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])
def load_img(filepath):
	img = Image.open(filepath)
	return img

class DatasetFromFolder(data.Dataset):
	def __init__(self, image_dir, input_transform=None, target_transform=None):
		super(DatasetFromFolder, self).__init__()
		self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

		self.input_transform = input_transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		input = load_img(self.image_filenames[index])
		target = input.copy()
		input = self.input_transform(input)
		target = self.target_transform(target)

		return input, target

	def __len__(self):
		return len(self.image_filenames)
###
def softmax_loss(logits, labels):
	logits = logits.permute(0, 2, 3, 1)
	logits = torch.reshape(logits, [-1, 256])
	labels = labels.to(torch.int64)
	labels = labels.permute(0, 2, 3, 1)
	labels = torch.reshape(labels, [-1])
	return F.cross_entropy(logits, labels)

def train(args, model, device, train_loader, optimizer, epoch):
	model.train()
	print('training')

	for batch_idx, (data, target) in enumerate(train_loader,1):
		data, target = data.to(device), target.to(device)	
		optimizer.zero_grad()
		conditioning_logits = model(lr_images = data)
		l2 = softmax_loss(conditioning_logits, torch.floor(target*255))
		loss = l2
		loss.backward()
		optimizer.step()
		if batch_idx % 10 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), (len(train_loader)*len(data)),
				100. * batch_idx / len(train_loader), loss.item()))
# 		if batch_idx%1000==0 :
# 			sample(model, data, target, len(data), mu=1.1, step=epoch*batch_idx)

def test(args, model, device, test_loader, epoch):
	model.eval()
	test_loss = 0
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(test_loader,1):
			data, target = data.to(device), target.to(device)
			conditioning_logits = model(lr_images = data)
			l2 = softmax_loss(conditioning_logits, torch.floor(target*255))
			test_loss += l2 # sum up batch loss
	test_loss /= len(test_loader)*len(data)
	print("test_loss : ", test_loss.item())
	sample(model, data, target, len(data), mu=1.1, step=epoch)


def logits_2_pixel_value(logits, mu=1.1):
	rebalance_logits = logits * mu
	probs = softmax(rebalance_logits)
	pixel_dict = torch.arange(0, 256, dtype=torch.float32).to("cuda")
	pixels = torch.sum(probs*pixel_dict, dim=1)
	return (pixels/255)

def softmax(x):
	a, b = torch.max(x, -1, keepdim=True, out=None)
	e_x = torch.exp(x - a)
	return e_x / e_x.sum(dim=-1, keepdim =True) # only difference

def sample(model, data, target, batch_size, mu=1.1, step=None):
	with torch.no_grad():
		np_lr_imgs = data
		np_hr_imgs = target
		c_logits = model.conditioning_network
		#p_logits = model.prior_network
		gen_hr_imgs = torch.zeros((batch_size, 3, 32, 32), dtype=torch.float32).to("cuda")
		np_c_logits = c_logits(np_lr_imgs)
		for i in range(32):
			for j in range(32):
				for c in range(3):
					new_pixel = logits_2_pixel_value(np_c_logits[:, c*256:(c+1)*256, i, j], mu=mu)
					gen_hr_imgs[:, c, i, j] = new_pixel
		samples_dir =  "/home/eee/ug/15084005/DIH/samples_ip/"
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
	parser.add_argument("--num_epoch", type = int, default = 60, help = "no of epoch")
	parser.add_argument("--batch_size", type = int, default = 64, help = "batch size")
	parser.add_argument("--learning_rate", type = float, default = 4e-4, help = "learning rate")
	args = parser.parse_args()

	use_cuda = args.use_gpu and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	torch.cuda.set_device(0)

	image_dir = '/home/eee/ug/15084005/DIH/CelebA/CelebA/train/img_align_celeba/'
	
	train_set = DatasetFromFolder(image_dir,input_transform=input_transform(1),target_transform=target_transform(1))
	train_loader = DataLoader(dataset=train_set, batch_size = args.batch_size, shuffle=True)
	
	image_dir_2 = '/home/eee/ug/15084005/DIH/CelebA/CelebA/test/data/'
	
	test_set = DatasetFromFolder(image_dir_2,input_transform=input_transform(1),target_transform=target_transform(1))
	test_loader = DataLoader(dataset=test_set, batch_size = args.batch_size, shuffle=True)
	
	model = Net()
	model = model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
	for epoch in range(1, args.num_epoch + 1):
		train(args, model, device, train_loader, optimizer, epoch)
		torch.save(model.state_dict(), "/home/eee/ug/15084005/DIH/models/"+str(epoch)+".pt")
		torch.save(model, "/home/eee/ug/15084005/DIH/model/"+str(epoch)+".pt")
		# sample(model, train_loader_lr, train_loader_hr, mu=1.1, step=epoch)
		test(args, model, device, test_loader, epoch)

if __name__ == '__main__':
	main()
