from train_clf import *
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# dataset for vehicle detection training
car_img_path = './dataset/vehicles/*/*.png'
notcar_img_path = './dataset/non-vehicles/*/*.png'

# settings for feature extraction
color_space = 'YCrCb'
spatial = 32
histbin = 32
hog_ornt = 9
hog_px_pc = 8
hog_cell_pb = 2

# Enable training if not done already
en_train = False

#save_model_name = 'tuned_svm.p'
save_model_name = 'lin_svc.p'
if en_train:
	X_scaler, X_train, X_test, y_train, y_test = data_prep(car_img_path, notcar_img_path, 
								color_space, spatial, histbin, 
								hog_ornt, hog_px_pc, hog_cell_pb)
	parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 20]}
	t=time.time()
	#model = train_tuned_svm(X_train, X_test, y_train, y_test, parameters)
	model = train_linear_svc(X_train, X_test, y_train, y_test)
	print(round(time.time()-t, 2), 'Seconds to train the SVM ...')
	print('Validation Accuracy of the SVM = ', round(model.score(X_test, y_test), 4))
	
	sav_dat = {'X_scaler': X_scaler, 'model': model}
	pickle.dump(sav_dat, open(save_model_name, 'wb'))
	
	#pdata = pickle.load(open(save_model_name, 'rb'))
	#X_scaler = pdata['X_scaler']
	#ld_model = pdata['model']
	#n_predict = 100
	#y_pred = ld_model.predict(X_test[0:n_predict])
	#y_diff = y_pred - y_test[0:n_predict]
	#err_rate = np.sum(np.absolute(y_diff))/n_predict
	##print('My SVC predicts: ', y_pred)
	##print('For these',n_predict, 'labels: ', y_test[0:n_predict])
	#print('Differences between actual and predicted: ', y_diff)
	#print('Prediction error rate: ', err_rate)
	#print(round(time.time()-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

else:
	pdata = pickle.load(open(save_model_name, 'rb'))
	X_scaler = pdata['X_scaler']
	model = pdata['model']

ystart = 400
ystop = 670
ht_thr = 4 

## for single image test
img_path = './test_images/test5.jpg'
img = mpimg.imread(img_path)
#scales = [0.8, 1.2, 1.5, 2]
scales = [1.2, 1.5]

def search_cars(img, scales, ystart, ystop, model, X_scaler, color_space, spatial, histbin, hog_ornt, hog_px_pc, hog_cell_pb, ht_thr):
	bbox=[]
	for scale in scales:
		bbox_list = find_cars(img, ystart, ystop, scale, model, X_scaler, color_space, (spatial, spatial), histbin, hog_ornt, hog_px_pc, hog_cell_pb)
		bbox=bbox+bbox_list
		bbox_list = find_cars(img, ystart, ystop, scale, model, X_scaler, color_space, (spatial, spatial), histbin, hog_ornt, hog_px_pc, hog_cell_pb)
		bbox=bbox+bbox_list
		bbox_list = find_cars(img, ystart, ystop, scale, model, X_scaler, color_space, (spatial, spatial), histbin, hog_ornt, hog_px_pc, hog_cell_pb)
		bbox=bbox+bbox_list
		bbox_list = find_cars(img, ystart, ystop, scale, model, X_scaler, color_space, (spatial, spatial), histbin, hog_ornt, hog_px_pc, hog_cell_pb)
		bbox=bbox+bbox_list

	heat_raw, heat_filt, draw_img = filter_cars(img, bbox, ht_thr)
	return heat_raw, heat_filt, draw_img

#heatraw, heatmap, draw_img = search_cars(img, scales, ystart, ystop, model, X_scaler, color_space, spatial, histbin, hog_ornt, hog_px_pc, hog_cell_pb, ht_thr)
#plt.figure()
#plt.imshow(np.array(draw_img*255, dtype=np.int))
#fig=plt.figure()
#ax1=fig.add_subplot(1,2,1)
#ax1.imshow(heatraw)
#ax1.set_title("Raw Heatmap")
#ax2=fig.add_subplot(1,2,2)
#ax2.set_title("Thresholded Heatmap")
#ax2.imshow(heatmap)
#plt.show()

# for video recording
from moviepy.editor import VideoFileClip
def proc_video(image):
	heatraw, heatmap, draw_img = search_cars(image, scales, ystart, ystop, model, X_scaler, color_space, spatial, histbin, hog_ornt, hog_px_pc, hog_cell_pb, ht_thr)
	return draw_img

#video_path = 'test_video.mp4'
#video_out = 'test_output.mp4'
video_path = 'project_video.mp4'
video_out = 'project_output.mp4'
clip = VideoFileClip(video_path)
proc_clip = clip.fl_image(proc_video)
proc_clip.write_videofile(video_out, audio=False)
