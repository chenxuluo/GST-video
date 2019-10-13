import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

__all__ = ['ResNet', 'resnet50', 'resnet101','resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, alpha, beta, stride = 1, downsample = None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm3d(planes)
		self.conv2 = nn.Conv3d(planes// beta, planes//alpha*(alpha-1), kernel_size=(1,3,3), stride=(1,stride,stride),
							   padding=(0,1,1), bias=False)
		self.Tconv = nn.Conv3d(planes//beta, planes//alpha, kernel_size = 3, bias = False,stride=(1,stride,stride), 
								padding = (1,1,1))
		self.bn2 = nn.BatchNorm3d(planes)
		self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm3d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride
		self.alpha = alpha
		self.beta = beta


	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)


		if self.beta == 2:
			nchannels = out.size()[1] // self.beta 
			left  = out[:,:nchannels]
			right = out[:,nchannels:]

			out1 = self.conv2(left)
			out2 = self.Tconv(right)
			
		else:
			out1 = self.conv2(out)
			out2 = self.Tconv(out)


		out = torch.cat((out1,out2),dim=1)
		out = self.bn2(out)
		out = self.relu(out)
		
		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(residual)

		out += residual
		out = self.relu(out)

		return out


class ResNet(nn.Module):

	def __init__(self, block, layers, alpha, beta, num_classes=1000):
		self.inplanes = 64
		super(ResNet, self).__init__()
		self.conv1 = nn.Conv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3),
							   bias=False)
		self.bn1 = nn.BatchNorm3d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
		self.layer1 = self._make_layer(block, 64, layers[0], alpha, beta)
		self.layer2 = self._make_layer(block, 128, layers[1], alpha, beta, stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], alpha, beta, stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], alpha, beta, stride=2)
		self.avgpool = nn.AvgPool2d(7, stride=1)
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm3d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, alpha, beta, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv3d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=(1,stride,stride), bias=False),
				nn.BatchNorm3d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, alpha, beta, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, alpha, beta))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.transpose(1,2).contiguous()
		x = x.view((-1,)+x.size()[2:])

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


def resnet50(alpha, beta,**kwargs):
	"""Constructs a ResNet-50 based model.
	"""
	model = ResNet(Bottleneck, [3, 4, 6, 3], alpha, beta, **kwargs)
	checkpoint = model_zoo.load_url(model_urls['resnet50'])
	layer_name = list(checkpoint.keys())
	for ln in layer_name:
		if 'conv' in ln or 'downsample.0.weight' in ln:
			checkpoint[ln] = checkpoint[ln].unsqueeze(2)
		if 'conv2' in ln:
			n_out, n_in, _, _, _ = checkpoint[ln].size()
			checkpoint[ln] = checkpoint[ln][:n_out // alpha * (alpha - 1), :n_in//beta,:,:,:]
	model.load_state_dict(checkpoint,strict = False)

	return model


def resnet101(alpha, beta ,**kwargs):
	"""Constructs a ResNet-101 model.
	Args:
		groups
	"""
	model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
	checkpoint = model_zoo.load_url(model_urls['resnet101'])
	layer_name = list(checkpoint.keys())
	for ln in layer_name:
		if 'conv' in ln or 'downsample.0.weight' in ln:
			checkpoint[ln] = checkpoint[ln].unsqueeze(2)
		if 'conv2' in ln:
			n_out, n_in, _, _, _ = checkpoint[ln].size()
			checkpoint[ln] = checkpoint[ln][:n_out // alpha * (alpha - 1), :n_in//beta,:,:,:]
	model.load_state_dict(checkpoint,strict = False)

	return model
