import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
	expansion = 1
	__constants__ = ['downsample']

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
				 base_width=64, dilation=1, norm_layer=None):
		super(BasicBlock, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		if groups != 1 or base_width != 64:
			raise ValueError('BasicBlock only supports groups=1 and base_width=64')
		if dilation > 1:
			raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
		# Both self.conv1 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4
	__constants__ = ['downsample']

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
				 base_width=64, dilation=1, norm_layer=None):
		super(Bottleneck, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		width = int(planes * (base_width / 64.)) * groups
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width)
		self.conv2 = conv3x3(width, width, stride, groups, dilation)
		self.bn2 = norm_layer(width)
		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class AUNet(nn.Module):

	def __init__(self, block=BasicBlock, layers=[3, 4, 6, 3], num_classes=14, zero_init_residual=True,
				 groups=1, width_per_group=64, replace_stride_with_dilation=None,
				 norm_layer=None, flag=False):
		super(AUNet, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer
		self.num_classes = num_classes

		self.inplanes = 64
		self.dilation = 1
		if replace_stride_with_dilation is None:
			# each element in the tuple indicates if we should replace
			# the 2x2 stride with a dilated convolution instead
			replace_stride_with_dilation = [False, False, False]
		if len(replace_stride_with_dilation) != 3:
			raise ValueError("replace_stride_with_dilation should be None "
							 "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
		self.groups = groups
		self.base_width = width_per_group
		self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = norm_layer(self.inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
									   dilate=replace_stride_with_dilation[0])
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
									   dilate=replace_stride_with_dilation[1])
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
									   dilate=replace_stride_with_dilation[2])


		#The depth of decoder part could be adjusted flexible.
		self.inplanes += 256
		self.layer5 = self._make_layer(block, 256, blocks=3)
		self.inplanes += 128
		self.layer6 = self._make_layer(block, 128, blocks=3)

		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

		self.atttention_weight = nn.ModuleList()
		self.softmax = nn.ModuleList()
		self.fc = nn.ModuleList()
		for i in range(self.num_classes):
			self.atttention_weight.append(nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1)))
			self.softmax.append(nn.Softmax(dim=1))
			self.fc.append(nn.Linear(128, 1))

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
		norm_layer = self._norm_layer
		downsample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
							self.base_width, previous_dilation, norm_layer))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups=self.groups,
								base_width=self.base_width, dilation=self.dilation,
								norm_layer=norm_layer))

		return nn.Sequential(*layers)

	def _forward_impl(self, x, lung, return_weight, return_cam):
		# See note [TorchScript super()]
		inputs = x

		x = self.conv1(x)
		x = self.bn1(x)
		x1 = self.relu(x)
		x = self.maxpool(x1)

		x2 = self.layer1(x)
		x3 = self.layer2(x2)
		x4 = self.layer3(x3)
		x5 = self.layer4(x4)

		x5 = nn.UpsamplingBilinear2d(size=(x4.size(2), x4.size(3)))(x5)
		x4 = torch.cat((x4, x5), dim=1)
		x4 = self.layer5(x4)

		x4 = nn.UpsamplingBilinear2d(size=(x3.size(2), x3.size(3)))(x4)
		x3 = torch.cat((x3, x4), dim=1)
		x3 = self.layer6(x3)

		_A = x3
		lung = F.interpolate(lung, size=(_A.size(2), _A.size(3)), mode='bilinear', align_corners=False)

		A = _A.permute(0,2,3,1).contiguous()
		h = A.size(1)
		w = A.size(2)
		A = A.view(-1, h*w, 128)
		weight = A.view(-1, 128)

		for i in range(self.num_classes):
			_x = self.atttention_weight[i](weight).view(-1, h*w, 1)
			_x = self.softmax[i](_x)
			_y = _x.view(-1, 1, h, w)
			_y = torch.mul(_y, lung)
			_y = _y / torch.sum(_y, dim=(1,2,3), keepdim=True)
			_x = _y.view(-1, h*w, 1)
			if i > 0:
				weight_list = torch.cat((weight_list, _y), dim=1)
			else:
				weight_list = _y
			_x = torch.mul(A, _x)
			_x = torch.sum(_x, dim=1)
			_x = self.fc[i](_x)
			if i > 0:
				output = torch.cat((output, _x), dim=1)
			else:
				output = _x
		raw_weight_list = weight_list
		weight_list = nn.UpsamplingBilinear2d(size=(inputs.size(2), inputs.size(3)))(weight_list)
		weight_list /= torch.sum(weight_list, dim=(2,3)).view(-1, self.num_classes, 1, 1)
		if return_weight:
			if return_cam:
				A = _A
				for i in range(self.num_classes):
					t = self.fc[i].weight[0].view(1, -1, 1, 1)
					t = torch.mul(A, t)
					t = torch.sum(t, dim=1)
					t = t.view(t.size(0), 1, t.size(1), t.size(2))
					t = torch.mul(t, raw_weight_list[:,i].unsqueeze(1))
					t = F.interpolate(t, size=(inputs.size(2), inputs.size(3)), mode='bilinear', align_corners=False)
					if i == 0:
						cam = t
					else:
						cam = torch.cat((cam, t), dim=1)
				return (output, weight_list, cam)
			else:
				return (output, weight_list)
		else:
			return output
	'''
	forward function
	Inputs:
	x: the chest x-ray image input (one channel).
	lung: the lung segmentation 0-1 mask, same size as x.
	return_weight: if true, the attention weight will be returned.
	return_cam: if true, the class activation mapping will be returned. 
		Note: this parameter is valid only if the return_weight is true. 

	Return:
	output: the prediction vector (batchsize, class_num=14) without softmax normalization
	weight_list: if return_weight is True, the attention weight (batchsize, class_num=14, input.H, input.W)
	cam: if return_weight is True, and return_cam is True, the class activation mapping (batchsize, class_num=14, input.H, input.W)
		The returned CAM is not applied with the ReLU function. (In our paper, the ReLU function is applied on the post-processing step)
	'''
	def forward(self, x, lung, return_weight=False, return_cam=False):
		return self._forward_impl(x, lung, return_weight, return_cam)



if __name__ == '__main__':
	#A toy input to validate the model
	Net = AUNet(flag=False)
	input_xray_image = torch.rand(2, 1, 256, 383)
	input_lung_segmentation = torch.ones(2, 1, 256, 383)
	input_lung_segmentation[:,:,1] = 0
	output, weight, cam = Net(input_xray_image, input_lung_segmentation, True, True)
	print(output.size())
	print(weight.size())
	print(torch.sum(weight, dim=(2,3)))
	print(cam.size())

		