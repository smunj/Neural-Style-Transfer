import cv2
import numpy as np
from time import time
from keras import backend as K
from keras.applications import vgg16
from scipy.optimize import fmin_l_bfgs_b

epochs = 15
alpha, beta, gamma = 0.025, 1.0, 1.5
content_img_path = 'contents/virat.jpg'
style_img_path = 'styles/fire.jpg'

content_name = content_img_path[9:].split('.')[0]
style_name = style_img_path[7:].split('.')[0]

def preprocess(img_path, shape=None):
	img = cv2.imread(img_path)
	if shape:
		img = cv2.resize(img, shape)
	img = img.astype('float32')
	img[:, :, 0] -= 103.939
	img[:, :, 1] -= 116.779
	img[:, :, 2] -= 123.68
	img = np.expand_dims(img, axis=0)
	return img

def deprocess(img, mode='rgb'):
	assert(img.ndim in [3, 4])
	if img.ndim == 4:
		img = np.squeeze(img, axis=0)
	img[:, :, 0] += 103.939
	img[:, :, 1] += 116.779
	img[:, :, 2] += 123.68
	if mode == 'rgb':
		img = img[:, :, ::-1]
	img = np.clip(img, 0, 255).astype('uint8')
	return img

content_img = preprocess(content_img_path)
nrows, ncols = content_img.shape[1:-1]
style_img = preprocess(style_img_path, (ncols, nrows))

# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(121)
# ax.axis('off')
# ax.imshow(deprocess(content_img))
# ax = fig.add_subplot(122)
# ax.axis('off')
# ax.imshow(deprocess(style_img))
# plt.show()

content_placeholder = K.variable(content_img)
style_placeholder = K.variable(style_img)
combination_placeholder = K.placeholder(content_img.shape)

input_tensor = K.concatenate([content_placeholder, 
															style_placeholder, 
															combination_placeholder], 
															axis=0)

model = vgg16.VGG16(input_tensor=input_tensor, 
										weights='imagenet', 
										include_top=False)


def content_loss(content, combination):
	loss = K.sum(K.square(content - combination))
	return loss

def gram_matrix(img):
	assert(K.ndim(img) == 3)
	features = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))
	gram = K.dot(features, K.transpose(features))
	return gram

def style_loss(style, combination):
	assert(K.ndim(style) == 3)
	assert(K.ndim(combination) == 3)
	S = gram_matrix(style)
	C = gram_matrix(combination)
	size = ncols * nrows
	loss = K.sum(K.square(S - C)) / ((6 * size) ** 2)
	return loss

def total_variation_loss(img):
	assert(K.ndim(img) == 4)
	a = K.square(img[:, :nrows - 1, :ncols - 1, :] - img[:, 1:, :ncols - 1, :])
	b = K.square(img[:, :nrows - 1, :ncols - 1, :] - img[:, :nrows - 1, 1:, :])
	return K.sum(K.pow(a + b, 1.25))

def loss_tensor(content_layer, style_layers):
	loss = K.variable(0.)
	
	content_layer_output = model.get_layer(content_layer).output
	content_features = content_layer_output[0, :, :, :]
	combination_features = content_layer_output[2, :, :, :]
	loss += alpha * content_loss(content_features, combination_features)

	for layer_name in style_layers:
		style_layer_output = model.get_layer(layer_name).output
		style_features = style_layer_output[1, :, :, :]
		combination_features = style_layer_output[2, :, :, :]
		loss += beta * style_loss(style_features, combination_features) / len(style_layers)

	loss += gamma * total_variation_loss(combination_placeholder)
	
	return loss

content_layer = 'block5_conv2'
style_layers = ['block1_conv1', 'block2_conv1', 
								'block3_conv1', 'block4_conv1', 
								'block5_conv1']
loss = loss_tensor(content_layer, style_layers)
grads = K.gradients(loss, combination_placeholder)

outputs = [loss] + grads
f_outputs = K.function(inputs=[combination_placeholder, ], outputs=outputs)

class Evaluator:
	def __init__(self):
		self.loss_values, self.grad_values = None, None

	def eval_loss_grads(self, x):
		x = x.reshape((1, nrows, ncols, 3))
		outs = f_outputs([x])
		loss = outs[0]
		if len(outs[1:]) == 1:
			grads = outs[1].flatten().astype('float64')
		else:
			grads = np.array(outs[1:], dtype='float64').flatten()
		return loss, grads

	def loss(self, x):
		assert(self.loss_values is None)
		self.loss_values, self.grad_values = self.eval_loss_grads(x)
		return self.loss_values

	def grads(self, x):
		assert(self.loss_values is not None)
		g = np.copy(self.grad_values)
		self.loss_values, self.grad_values = None, None
		return g

e = Evaluator()
x = content_img.copy()
for i in range(epochs):
	print('\rEpoch {0:>3}'.format('#' + str(i + 1)), end=' ')
	start_time = time()
	x, min_val, info = fmin_l_bfgs_b(e.loss, x.flatten(), 
																	 fprime=e.grads, maxfun=20)
	print('Current Loss: {0:10.3e}'.format(min_val), end=' ')
	x = x.reshape((1, nrows, ncols, 3))
	img = deprocess(x.copy(), mode='bgr')
	fname = '{0}_{1}_epoch_{2}.jpg'.format(content_name, style_name, i + 1)
	cv2.imwrite(fname, img)
	end_time = time()
	print('Time taken: {0:6.2f}s'.format(end_time - start_time))

