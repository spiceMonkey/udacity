from util import *

import numpy as np
import glob
from sklearn import svm, tree, grid_search
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Read in car and non-car images
def data_prep(car_img_path, notcar_img_path, clr_space, spatial, histbin, hog_ornt, hog_px_pc, hog_cell_pb):
	cars = []
	notcars = []
	cars_images = glob.glob(car_img_path)
	notcars_images = glob.glob(notcar_img_path)
	for image in cars_images:
		cars.append(image)
	for image in notcars_images:
		notcars.append(image)
	
	car_features = extract_features(cars, cspace=clr_space, spatial_size=(spatial, spatial), hist_bins=histbin, orient=hog_ornt, 
					pix_per_cell=hog_px_pc, cell_per_block=hog_cell_pb)
	notcar_features = extract_features(notcars, cspace=clr_space, spatial_size=(spatial, spatial), hist_bins=histbin, orient=hog_ornt, 
					pix_per_cell=hog_px_pc, cell_per_block=hog_cell_pb)

	# Create an array stack of feature vectors
	X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
	# Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)
	
	# Define the labels vector
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
	
	
	# Split up data into randomized training and test sets
	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(
	    scaled_X, y, test_size=0.2, random_state=rand_state)

	print('Feature vector length:', len(X_train[0]))
	return X_scaler, X_train, X_test, y_train, y_test

def train_linear_svc(X_train, X_test, y_train, y_test):
	# Use a linear SVC 
	model = svm.LinearSVC()
	model.fit(X_train, y_train)
	
	return model

def train_decision_tree(X_train, X_test, y_train, y_test):
	# Use a linear SVC 
	model = tree.DecisionTreeClassifier()
	model.fit(X_train, y_train)
	
	return model

def train_tuned_svm(X_train, X_test, y_train, y_test, parameters):
	svr = svm.SVC()
	model = grid_search.GridSearchCV(svr, parameters)
	model.fit(X_train, y_train)
	
	print('Optimized model parameters: ', model.best_params_)

	return model
	
