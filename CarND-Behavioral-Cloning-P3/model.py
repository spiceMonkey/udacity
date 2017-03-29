import cv2
import numpy as np
from sklearn.utils import shuffle

# utility functions
def bgr2rgb(img):
	b = img[:,:,0]
	g = img[:,:,1]
	r = img[:,:,2]

	return np.dstack((r,g,b))

def proc_img(img):
	# cropping
	img_shape = img.shape
	img_out = img[60:img_shape[0]-20, :]
	img_shape = img_out.shape
	new_dim = (int(img_shape[1]/2), int(img_shape[0]/2))
	img_out = cv2.resize(img_out, new_dim, interpolation=cv2.INTER_AREA)
	img_out = img_out/255-0.5
	return img_out

# generate batch data due to memory limitation	
def gen_batch(samples, start_indx, batch_size=32):

	batch_samples = samples[start_indx:start_indx+batch_size]

	images = []
	angles = []
	correction = 0.25
	
	for batch_sample in batch_samples:

		src_path_cntr = batch_sample[0]	
		fname_cntr = src_path_cntr.split('/')[-1]
		cur_path_cntr = img_path+fname_cntr
		img_cntr = bgr2rgb(cv2.imread(cur_path_cntr))
		proc_img_cntr = proc_img(img_cntr)
		images.append(proc_img_cntr)
		str_cntr =  float(batch_sample[3])
		angles.append(str_cntr)
		# add flipped version
		images.append(np.fliplr(proc_img_cntr))
		angles.append(-str_cntr)

		src_path_left = batch_sample[1]	
		fname_left = src_path_left.split('/')[-1]
		cur_path_left = img_path+fname_left
		img_left = bgr2rgb(cv2.imread(cur_path_left))
		proc_img_left = proc_img(img_left)
		images.append(proc_img_left)
		str_left = str_cntr + correction
		angles.append(str_left)
		# add flipped version
		images.append(np.fliplr(proc_img_left))
		angles.append(-str_left)

		src_path_right = batch_sample[2]	
		fname_right = src_path_right.split('/')[-1]
		cur_path_right = img_path+fname_right
		img_right = bgr2rgb(cv2.imread(cur_path_right))
		proc_img_right = proc_img(img_right)
		images.append(proc_img_right)
		str_right = str_cntr - correction
		angles.append(str_right)
		# add flipped version
		images.append(np.fliplr(proc_img_right))
		angles.append(-str_right)
	
	X = np.array(images)
	y = np.array(angles)
	
	return shuffle(X, y)			
			
img_shape = (40, 160, 3)

# CNN architecture
from keras.models import Sequential
from keras.layers import Lambda, Dense, Dropout, Flatten, Activation
from keras.layers import Cropping2D, Conv2D, MaxPooling2D

model = Sequential()
k_size, s_size=5,2
model.add(Conv2D(filters=32, kernel_size=(k_size, k_size), strides=(s_size, s_size), padding='valid', use_bias=True, activation='relu', input_shape=img_shape))
model.add(Conv2D(filters=32, kernel_size=(k_size, k_size), strides=(s_size, s_size), padding="valid", use_bias=True, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="valid"))
model.add(Dropout(0.5))

k_size, s_size=3,1
model.add(Conv2D(filters=64, kernel_size=(k_size, k_size), strides=(s_size, s_size), padding="valid", use_bias=True, activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(k_size, k_size), strides=(s_size, s_size), padding="valid", use_bias=True, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="valid"))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))

# optimizer
model.compile(loss='mse', optimizer='adam')
# print out model summary
model.summary()

# load dataset
import csv
data_path = './car.data/'
log_path = data_path + 'driving_log.csv'
img_path = data_path + 'IMG/'

samples = []
with open(log_path) as logfile:
	text = csv.reader(logfile)
	for line in text:
		samples.append(line)

# split into training and validation data
from sklearn.model_selection import train_test_split
train_samples, valid_samples = train_test_split(shuffle(samples), test_size=0.2)

# start epoch and batch optimization
EPOCHS = 25
BATCH_SIZE = 64
TRAIN_BATCHES = len(train_samples)//BATCH_SIZE
VALID_BATCHES = len(valid_samples)//BATCH_SIZE

train_loss = []
valid_loss = []

# start optimization
import time
t0 = time.time()
for e in range(EPOCHS):

	t1 = time.time()
	shuffle(train_samples)
	shuffle(valid_samples)
	train_batch_loss = []	
	valid_batch_loss = []	

	print("Epoch = {}...".format(e+1))

	for tb in range(TRAIN_BATCHES):
		start = tb * BATCH_SIZE
		(X_train, y_train) = gen_batch(train_samples, start, BATCH_SIZE) 
		train_batch_loss.append(model.train_on_batch(X_train, y_train))

	for vb in range(VALID_BATCHES):
		start = vb * BATCH_SIZE
		(X_valid, y_valid) = gen_batch(valid_samples, start, BATCH_SIZE)
		valid_batch_loss.append(model.test_on_batch(X_valid, y_valid))

	train_loss.append(sum(train_batch_loss)/len(train_batch_loss))
	valid_loss.append(sum(valid_batch_loss)/len(valid_batch_loss))
	
	print("Train Loss = {0:.4f}, Valid Loss = {1:.4f}".format(train_loss[e], valid_loss[e]))
	model.save('model_my_batch_ep{}.h5'.format(e+1))

	print("Time for this epoch  = {0:.2f}seconds:".format(time.time()-t1))

print("Total time elapsed {0:.2f} seconds:".format(time.time()-t0))

import matplotlib.pyplot as plt
plt.plot(train_loss)
plt.plot(valid_loss)
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
