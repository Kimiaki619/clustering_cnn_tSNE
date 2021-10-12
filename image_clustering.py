# coding: utf-8

from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing import image
from keras.models import model_from_json,load_model
import tensorflow as tf

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import sys
import cv2
import os
from progressbar import ProgressBar 
import shutil
import pathlib

#たくぼがつくったものグラフ化するライブラリ
import tsne
import ArcFaceModel

#gpuを使用するためのコード
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto(
#    gpu_options=tf.GPUOptions(
#        visible_device_list="0", # specify GPU number
#        allow_growth=True
#    )
#)
#set_session(tf.Session(config=config))

DATA_DIR = '../data/'
VIDEOS_DIR = '../data/video/'                        # The place to put the video
TARGET_IMAGES_DIR = '../data/images/target/'         # The place to put the images which you want to execute clustering
CLUSTERED_IMAGES_DIR = '../data/images/clustered/'   # The place to put the images which are clustered
IMAGE_LABEL_FILE ='image_label.csv'                  # Image name and its label
IMAGE_CLASTER_FILE = 'label.csv'					#ラベルのcsvの名前をここに書く


class Image_Clustering:
	def __init__(self, n_clusters=2, video_file='IMG_2140.MOV', image_file_temp='img_%s.jpg', input_video=False,image_size=224, model_json=False,weight=False,path_model= './model'):
		self.n_clusters = n_clusters            # The number of cluster
		self.video_file = video_file            # Input video file name
		self.image_file_temp = image_file_temp  # Image file name template
		self.input_video = input_video          # If input data is a video
		self.model = self.model_vgg(model_json=model_json,weight=weight)
		self.model_json = model_json
		self.image_size = image_size
		self.path_model = path_model
		self.model.summary()

	def main(self):
		if self.input_video == True:
			self.video_2_frames()
		self.label_images()
		self.classify_images()
		

	def video_2_frames(self):
		print('Video to frames...')
		cap = cv2.VideoCapture(VIDEOS_DIR+self.video_file)

		# Remove and make a directory.
		if os.path.exists(TARGET_IMAGES_DIR):
			shutil.rmtree(TARGET_IMAGES_DIR)  # Delete an entire directory tree
		if not os.path.exists(TARGET_IMAGES_DIR):
			os.makedirs(TARGET_IMAGES_DIR)	# Make a directory

		i = 0
		while(cap.isOpened()):
			flag, frame = cap.read()  # Capture frame-by-frame
			if flag == False:
				break  # A frame is not left
			cv2.imwrite(TARGET_IMAGES_DIR+self.image_file_temp % str(i).zfill(6), frame)  # Save a frame
			i += 1
			print('Save', TARGET_IMAGES_DIR+self.image_file_temp % str(i).zfill(6))

		cap.release()  # When everything done, release the capture
		print('')

	#どのモデルを使うかの関数
	def model_vgg(self,model_json,weight):
		if model_json == False:
			model = VGG19(weights='imagenet', include_top=False)
			
		else:
			#model = ArcFaceModel.arcface_model(classes=self.n_clusters,alpha_=0.5,input_shape=(224,224,3),path=model_json,weight= weight)
			#model = model.train_VGG19()
            
			model = load_model(weight,compile=False)
			#model = model_from_json(model_json)
			#model.load_weights(os.path.join(self.path_model,weight))
			
		return model


	def label_images(self):
		print('Label images...')	
		# Get images
		#クラスタリングした画像のパス
		path = pathlib.Path(TARGET_IMAGES_DIR).glob('*.jpg')
		path_l = [p for p in path]
		#クラスタリングする画像
		images = [f for f in os.listdir(TARGET_IMAGES_DIR) if f[-4:] in ['.jpg']]
		assert(len(images)>0)
		
		X = []
		pb = ProgressBar(max_value=len(images))
		for i in range(len(images)):
			# Extract image features
			feat = self.__feature_extraction(TARGET_IMAGES_DIR+images[i])
			X.append(feat)
			pb.update(i)  # Update progressbar

		# Clutering images by k-means++
		X = np.array(X)
		Tsne = tsne.tSNE(X,path_l)
        #この関数にtsneの値が入っている
		X = Tsne.t_sne
		X = np.array(X)

		kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(X)
		print('')
		print('labels:')
		print(kmeans.labels_)
		print('')
		
		# Merge images and labels
		df = pd.DataFrame({'image': images, 'label': kmeans.labels_})
		df.to_csv(DATA_DIR+IMAGE_LABEL_FILE, index=False)

		#ラベルのcsv
		df_label = pd.read_csv(DATA_DIR+IMAGE_CLASTER_FILE,index_col=0)
		df_cla = pd.read_csv(DATA_DIR+IMAGE_LABEL_FILE,index_col=1)
		#ラベルの列を変える
		df_cla = df_cla.values
		df_cla = [df_cla[f][0] for f in range(len(df_cla))]
		df_label = df_label.reindex(df_cla)
		df_label_index = df_label.values
		df_label_index = [df_label_index[i][0] for i in range(len(df_label_index))]
        
		#df_label_index = [df_label_index[f][1] for f in range(len(df_label_index))]
		
		#print(df_cla)
        #df_cla = df_cla.reindex(index=df_label)
		#df_cla = df_cla.values
		#df_cla = [df_cla[f][0] for f in range(len(df_cla))]
		print(df_label_index)
		print(kmeans.labels_)
		print("---------------------------")
		
        
		#ここで画像つきのtsneを出力
		Tsne.graph_image()
		#ここではkmeansのクラス分けを見る
		Tsne.graph_clstering(kmeans.labels_)
		Tsne.graph_clstering_3d(kmeans.labels_)
		#ここでは元々のラベルを見る
		Tsne.graph_clstering(df_label_index)
		Tsne.graph_clstering_3d(df_label_index)
		

	def __feature_extraction(self, img_path):
		img = image.load_img(img_path, target_size=(self.image_size, self.image_size))  # resize
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)  # add a dimention of samples
		x = preprocess_input(x)  # RGB 2 BGR and zero-centering by mean pixel based on the position of channels
        
		if self.model_json == False:
			#self.model.summary()
			feat = self.model.predict(x)# Get image features
		else:
			#self.model.layers.pop()
			self.model.layers.pop()
			self.model.layers.pop()
			self.model.layers.pop()
			self.model.layers.pop()
			self.model.layers.pop()
			#self.model.summary()
			feat = self.model.predict(x)
		#print(feat)
		#if self.path == False:
		feat = feat.flatten()  # Convert 3-dimentional matrix to (1, n) array
		
		return feat


	def classify_images(self):
		print('Classify images...')

		# Get labels and images
		df = pd.read_csv(DATA_DIR+IMAGE_LABEL_FILE)
		labels = list(set(df['label'].values))
		
		# Delete images which were clustered before
		if os.path.exists(CLUSTERED_IMAGES_DIR):
			shutil.rmtree(CLUSTERED_IMAGES_DIR)

		for label in labels:
			print('Copy and paste label %s images.' % label)

			# Make directories named each label
			new_dir = CLUSTERED_IMAGES_DIR + str(label) + '/'
			if not os.path.exists(new_dir):
				os.makedirs(new_dir)

			# Copy images to the directories
			clustered_images = df[df['label']==label]['image'].values
			for ci in clustered_images:
				src = TARGET_IMAGES_DIR + ci
				dst = CLUSTERED_IMAGES_DIR + str(label) + '/' + ci
				shutil.copyfile(src, dst)

		
if __name__ == "__main__":
	Image_Clustering().main()