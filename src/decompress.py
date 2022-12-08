import os
import numpy as np
import math

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from data_utils import SequenceGenerator, data_padding

from scipy import sparse
from scipy.sparse import *

from numba import cuda

import zstd
from PIL import Image

import time


def finding_difference(arr):
	arr_f = arr.flatten()
	tmp = arr_f[0]
	for i in range(1, len(arr_f)):
			arr_f[i] = tmp - arr_f[i]
			tmp = arr_f[i]
	
	return arr_f.reshape(arr.shape)

def replacing_based_on_frequency(arr, table, xp):
	result = arr.copy()
	for idx, num in enumerate(table):
		result = xp.where(result == idx, num, result)
	
	return result


def run(WEIGHTS_DIR, DATA_DIR, OUTPUT_DIR, GPU_FLAG, VERBOSE):
	if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
	
	batch_size = 10

	weights_file = os.path.join(WEIGHTS_DIR, 'prednet_weights.hdf5')
	json_file = os.path.join(WEIGHTS_DIR, 'prednet_model.json')

	try:
		with open(os.path.join(DATA_DIR, 'filename.txt'), 'r', encoding='UTF-8') as f:
			file_names = [s.strip() for s in f.readlines()]
	except FileNotFoundError as e:
		print("ERROR:No such file or directory:", os.path.join(DATA_DIR, 'filename.txt'))
		exit()

	# Load trained model
	try:
		f = open(json_file, 'r')
	except FileNotFoundError as e:
		print("ERROR: No such file or directory:", json_file)
		exit()
	else :
		json_string = f.read()
		f.close()
		train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
	try:
		train_model.load_weights(weights_file)
	except OSError as e:
		print("ERROR: No such file or directory:", weights_file)
		exit()

	# Create testing model (to output predictions)
	layer_config = train_model.layers[1].get_config()
	layer_config['output_mode'] = 'prediction'
	data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']

	# モデルセッティング（DWP)
	test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
	input_shape = list(train_model.layers[0].batch_input_shape[2:])
	input_shape.insert(0, None)
	inputs = Input(shape=tuple(input_shape))
	predictions = test_prednet(inputs)
	test_model = Model(inputs=inputs, outputs=predictions)

	try:
		with open(os.path.join(DATA_DIR, "key_frame.dat"), mode='rb') as f:
			input_data = zstd.decompress(f.read())
	except FileNotFoundError as e:
		print("ERROR: No such file or directory:", os.path.join(DATA_DIR, "key_frame.dat"))
		exit()
	
	X_test = np.frombuffer(input_data, dtype='uint8')

	try:
		with open(os.path.join(DATA_DIR, "entropy.dat"), mode='rb') as f:
			input_data = zstd.decompress(f.read())
	except FileNotFoundError as e:
		print("ERROR: No such file or directory:", os.path.join(DATA_DIR, "entropy.dat"))
		exit()
	
	decompress_data = np.frombuffer(input_data, dtype='int16')

	# warm up(PREPROCESS)の値を復元
	warm_up = decompress_data[-1]
	warm_up = warm_up.astype('int16')
	decompress_data = decompress_data[:-1]

	# 圧縮前のnumpyのshapeを復元
	X_test_shape = decompress_data[-5:]
	X_test_shape = X_test_shape.astype('int16')
	decompress_data = decompress_data[:-5]

	X_test = X_test.reshape(X_test_shape)

	X_test = X_test / 255

	# 推論用にキーフレームにパディング
	X_test_pad = data_padding(X_test)


	# キーフレームの場所と何枚先を推論したかを復元
	key_frame_check = []
	for X_testcut in X_test_pad:
		for idx in range(X_testcut.shape[0]):
			if not np.all(X_testcut[idx] == 0):
				key_frame_check.append(idx)

	key_frame_check.append(X_test_pad.shape[1])

	if test_model.input.shape[2] != X_test_pad.shape[2] or test_model.input.shape[3] != X_test_pad.shape[3]:
		print("ERROR:keyframe size and model size do not match.")
		print("model size: height ", test_model.input.shape[2] - 7, "～", test_model.input.shape[2], " width ", test_model.input.shape[3] - 7, "～", test_model.input.shape[3])
		print("key frame size: height ", X_test.shape[2], " width ",  X_test.shape[3])
		exit()


	# predict
	result_list = []
	key_idx = warm_up
	X_test_one = X_test_pad[0, 0]
	X_test_one = X_test_one[np.newaxis, np.newaxis, :, :, :]
	warm_up_frame = test_model.predict(X_test_one, batch_size)
	for _ in range(warm_up):
		result_list.append(warm_up_frame)

	for idx in range(warm_up, len(key_frame_check[warm_up:]) + warm_up - 1):
		for predict_idx in range(key_frame_check[idx], key_frame_check[idx+1]):
			if predict_idx == key_frame_check[idx]:
				X_test_one = X_test_pad[0, predict_idx]
				X_test_one = X_test_one[np.newaxis, np.newaxis, :, :, :]
				X_test_tmp = np.zeros(X_test_one.shape)
				X_test_one = np.hstack([X_test_one, X_test_tmp])
				X_hat = test_model.predict(X_test_one, batch_size)

				X_hat_predict_one = X_test_pad[0, predict_idx]
				X_hat_predict_one = X_hat_predict_one[np.newaxis, np.newaxis, :, :, :]
				result_list.append(X_hat_predict_one)
			
			elif predict_idx == key_frame_check[idx] + 1:
				X_test_one = X_test_pad[0, predict_idx - 1]
				X_test_one = X_test_one[np.newaxis, np.newaxis, :, :, :]
				X_test_tmp = np.zeros(X_test_one.shape)
				X_test_one = np.hstack([X_test_one, X_test_tmp])
				X_hat = test_model.predict(X_test_one, batch_size)

				X_hat_predict_one = X_hat[0, 1]
				X_hat_predict_one = X_hat_predict_one[np.newaxis, np.newaxis, :, :, :]
				result_list.append(X_hat_predict_one)

			else:
				X_test_one = result_list[-1]
				X_test_tmp = np.zeros(X_test_one.shape)
				X_test_one = np.hstack([X_test_one, X_test_tmp])
				X_hat = test_model.predict(X_test_one, batch_size)

				X_hat_predict_one = X_hat[0, 1]
				X_hat_predict_one = X_hat_predict_one[np.newaxis, np.newaxis, :, :, :]
				result_list.append(X_hat_predict_one)

	# 推論結果を一つにまとめる
	X_hat_flat = result_list[0]
	for X_hat_np in result_list[1:]:
		X_hat_flat = np.hstack([X_hat_flat, X_hat_np])

	X_hat_flat[0, 0] = X_test_pad[0, 0]

	# 推論結果からパディングを外す
	X_hat_no_pad = X_hat_flat[:,:, :X_test.shape[2], :X_test.shape[3]]

	# GPU無:numpy GPU有:cupyに設定
	if GPU_FLAG:
		# tensorflowが占有しているメモリを解放
		cuda.select_device(0)
		cuda.close()
		import cupy as xp
	else:
		import numpy as xp

	
	start = time.time()
	# エントロピー符号化のmapping tableを復元
	table = []
	table_len = decompress_data[-1]

	if table_len == -1:
		decompress_data = decompress_data[:-1]
	else:
		table_start = -table_len - 1
		table_np = decompress_data[table_start:-1]
		for num in table_np:
			table.append(num)

		table_xp = xp.array(table)

		elapsed_time = time.time() - start

		if VERBOSE: print ("table_create:{0}".format(elapsed_time) + "[sec]")

		# エントロピー符号化のmapping tableを配列から削除
		decompress_data = decompress_data[:table_start]

		# GPUがあるならばcupyに変換
		if GPU_FLAG:
			decompress_data = xp.asarray(decompress_data)

		start = time.time()
		# エントロピー符号化から復元
		decompress_data = replacing_based_on_frequency(decompress_data, table_xp, xp)
		
		elapsed_time = time.time() - start

		if VERBOSE: print ("replacing_based_on_frequency:{0}".format(elapsed_time) + "[sec]")

		# エントロピー符号化のテーブル作成のために1600との差分として保存したものの復元
		decompress_data = xp.subtract(1600, decompress_data)

	start = time.time()
	# Density-based Spatial Encoding
	tmp = xp.reshape(decompress_data, X_test_shape)
	
	# cupyに変換していたらnumpyに戻す
	if GPU_FLAG:
		tmp = xp.asnumpy(tmp)
	difference_first = finding_difference(tmp)

	elapsed_time = time.time() - start

	if VERBOSE: print ("finding_difference:{0}".format(elapsed_time) + "[sec]")

	# 画像の出力
	decompress = X_hat_no_pad * 255
	decompress = decompress - difference_first

	decompress = np.where(decompress > 255, 255, decompress)
	decompress = np.where(decompress < 0, 0, decompress)

	count = 0

	if len(file_names) != X_test_shape[1]:
		print("ERROR：The lengths of filename.txt and images do not match.")
		print("filename.txt：", len(file_names))
		print("number of images", X_test_shape[1])
		exit()

	for i in range(X_test_shape[0]):
		for j in range(X_test_shape[1]):
			tmp_np = decompress[i,j,:,:,:]
			tmp_np = tmp_np.astype('uint8')
			tmp_image = Image.fromarray(tmp_np)
			tmp_image.save(os.path.join(OUTPUT_DIR, file_names[count]))
			count += 1

def run_save_memory(WEIGHTS_DIR, DATA_DIR, OUTPUT_DIR, GPU_FLAG, VERBOSE, IMAGE_NUM_PER_TIME):
	if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
	
	batch_size = 10

	weights_file = os.path.join(WEIGHTS_DIR, 'prednet_weights.hdf5')
	json_file = os.path.join(WEIGHTS_DIR, 'prednet_model.json')

	try:
		with open(os.path.join(DATA_DIR, 'filename.txt'), 'r', encoding='UTF-8') as f:
			file_names = [s.strip() for s in f.readlines()]
	except FileNotFoundError as e:
		print("ERROR:No such file or directory:", os.path.join(DATA_DIR, 'filename.txt'))
		exit()

	# Load trained model
	try:
		f = open(json_file, 'r')
	except FileNotFoundError as e:
		print("ERROR: No such file or directory:", json_file)
		exit()
	else :
		json_string = f.read()
		f.close()
		train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
	try:
		train_model.load_weights(weights_file)
	except OSError as e:
		print("ERROR: No such file or directory:", weights_file)
		exit()

	# Create testing model (to output predictions)
	layer_config = train_model.layers[1].get_config()
	layer_config['output_mode'] = 'prediction'
	data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']

	# モデルセッティング（DWP)
	test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
	input_shape = list(train_model.layers[0].batch_input_shape[2:])
	input_shape.insert(0, None)
	inputs = Input(shape=tuple(input_shape))
	predictions = test_prednet(inputs)
	test_model = Model(inputs=inputs, outputs=predictions)

	try:
		with open(os.path.join(DATA_DIR, "key_frame.dat"), mode='rb') as f:
			input_data = zstd.decompress(f.read())
	except FileNotFoundError as e:
		print("ERROR: No such file or directory:", os.path.join(DATA_DIR, "key_frame.dat"))
		exit()
	
	X_test = np.frombuffer(input_data, dtype='uint8')

	try:
		with open(os.path.join(DATA_DIR, "entropy.dat"), mode='rb') as f:
			input_data = zstd.decompress(f.read())
	except FileNotFoundError as e:
		print("ERROR: No such file or directory:", os.path.join(DATA_DIR, "entropy.dat"))
		exit()
	
	decompress_data_all = np.frombuffer(input_data, dtype='int16')

	# warm up(PREPROCESS)の値を復元
	warm_up = decompress_data_all[-1]
	warm_up = warm_up.astype('int16')
	decompress_data_all = decompress_data_all[:-1]

	# 圧縮前のnumpyのshapeを復元
	X_test_shape = decompress_data_all[-5:]
	X_test_shape = X_test_shape.astype('int16')
	decompress_data_all = decompress_data_all[:-5]

	X_test = X_test.reshape(X_test_shape)

	X_test = X_test / 255

	# 推論用にキーフレームにパディング
	X_test_pad_all = data_padding(X_test)

	# キーフレームの場所と何枚先を推論したかを復元
	key_frame_check_all = []
	for X_testcut in X_test_pad_all:
		for idx in range(X_testcut.shape[0]):
			if not np.all(X_testcut[idx] == 0):
				key_frame_check_all.append(idx)

	key_frame_check_all.append(X_test_pad_all.shape[1])

	if test_model.input.shape[2] != X_test_pad_all.shape[2] or test_model.input.shape[3] != X_test_pad_all.shape[3]:
		print("ERROR:keyframe size and model size do not match.")
		print("model size: height ", test_model.input.shape[2] - 7, "～", test_model.input.shape[2], " width ", test_model.input.shape[3] - 7, "～", test_model.input.shape[3])
		print("key frame size: height ", X_test.shape[2], " width ",  X_test.shape[3])
		exit()

	# GPU無:numpy GPU有:cupyに設定
	if GPU_FLAG:
		# tensorflowが占有しているメモリを解放
		cuda.select_device(0)
		cuda.close()
		import cupy as xp
	else:
		import numpy as xp

	
	start = time.time()
	# エントロピー符号化のmapping tableを復元
	table = []
	table_len = decompress_data_all[-1]

	if table_len == -1:
		decompress_data_all = decompress_data_all[:-1]
	else:
		table_start = -table_len - 1
		table_np = decompress_data_all[table_start:-1]
		for num in table_np:
			table.append(num)

		table_xp = xp.array(table)

		elapsed_time = time.time() - start

		if VERBOSE: print ("table_create:{0}".format(elapsed_time) + "[sec]")

		# エントロピー符号化のmapping tableを配列から削除
		decompress_data_all = decompress_data_all[:table_start]

		# GPUがあるならばcupyに変換
		if GPU_FLAG:
			decompress_data_all = xp.asarray(decompress_data_all)

		start = time.time()
		# エントロピー符号化から復元
		decompress_data_all = replacing_based_on_frequency(decompress_data_all, table_xp, xp)
		
		elapsed_time = time.time() - start

		if VERBOSE: print ("replacing_based_on_frequency:{0}".format(elapsed_time) + "[sec]")

		# エントロピー符号化のテーブル作成のために1600との差分として保存したものの復元
		decompress_data_all = xp.subtract(1600, decompress_data_all)

	start = time.time()
	# Density-based Spatial Encoding
	decompress_data_all_reshaped = xp.reshape(decompress_data_all, X_test_shape)
	
	# cupyに変換していたらnumpyに戻す
	if GPU_FLAG:
		tmp = xp.asnumpy(decompress_data_all_reshaped)
	difference_first_all = finding_difference(decompress_data_all_reshaped)

	elapsed_time = time.time() - start

	if VERBOSE: print ("finding_difference:{0}".format(elapsed_time) + "[sec]")

	if len(file_names) != X_test_shape[1]:
		print("ERROR：The lengths of filename.txt and images do not match.")
		print("filename.txt：", len(file_names))
		print("number of images", X_test_shape[1])
		exit()

	# 逐次処理準備
	image_num = X_test_shape[1]
	# 各逐次処理の最初の画像indexを格納したlist作成(-iで指定した画像枚数を最も近いキーフレームを採用)
	image_index_li = list(range(0,image_num))
	sequential_first_key_frame_li = [0]
	count = 1
	while(len(image_index_li)!=0):
		if (len(image_index_li) > IMAGE_NUM_PER_TIME ):
			first_image_index1 = np.abs(np.asfarray(key_frame_check_all) - (IMAGE_NUM_PER_TIME*count)).argmin()
			# 各逐次処理の開始の画像
			first_image_index2 = key_frame_check_all[first_image_index1]
			sequential_first_key_frame_li.append(first_image_index2)
			image_index_li = image_index_li[image_index_li.index(first_image_index2):]
			count += 1
		else:
			image_index_li = []

	# 各逐次処理の最初と最後の画像indexを格納したlist作成
	sequential_key_frame_li = []
	for i in range(len(sequential_first_key_frame_li)):
		if i != sequential_first_key_frame_li.index(sequential_first_key_frame_li[-1]):
			sequential_key_frame_li.append((sequential_first_key_frame_li[i], sequential_first_key_frame_li[i+1]-1))
		else:
			sequential_key_frame_li.append((sequential_first_key_frame_li[i], image_num-1))

	print("Split execution mode.\n")

	# 逐次実行
	for sequential, index_tuple in enumerate(sequential_key_frame_li):
		sequential += 1
		start_image_index = index_tuple[0]
		end_image_index = index_tuple[1]

		print(f"Number of images for the {sequential} time：{end_image_index-start_image_index+1}")

		# key_frame
		X_test_pad = X_test_pad_all[:,start_image_index:end_image_index+1, :, :, :]
		key_frame_check = key_frame_check_all[key_frame_check_all.index(start_image_index):key_frame_check_all.index(end_image_index+1)+1]

		# entorpy
		difference_first = difference_first_all[:,start_image_index:end_image_index+1, :, :, :]

		# predict
		result_list = []
		key_idx = warm_up
		X_test_one = X_test_pad[0, 0]
		X_test_one = X_test_one[np.newaxis, np.newaxis, :, :, :]
		warm_up_frame = test_model.predict(X_test_one, batch_size)
		if sequential == 1 :
			for _ in range(warm_up):
				result_list.append(warm_up_frame)

			for idx in range(warm_up, len(key_frame_check[warm_up:]) + warm_up - 1):
				for predict_idx in range(key_frame_check[idx], key_frame_check[idx+1]):

					if predict_idx == key_frame_check[idx]:
						X_test_one = X_test_pad[0, predict_idx]
						X_test_one = X_test_one[np.newaxis, np.newaxis, :, :, :]
						X_test_tmp = np.zeros(X_test_one.shape)
						X_test_one = np.hstack([X_test_one, X_test_tmp])
						X_hat = test_model.predict(X_test_one, batch_size)

						X_hat_predict_one = X_test_pad[0, predict_idx]
						X_hat_predict_one = X_hat_predict_one[np.newaxis, np.newaxis, :, :, :]
						result_list.append(X_hat_predict_one)
					
					elif predict_idx == key_frame_check[idx] + 1:
						X_test_one = X_test_pad[0, predict_idx - 1]
						X_test_one = X_test_one[np.newaxis, np.newaxis, :, :, :]
						X_test_tmp = np.zeros(X_test_one.shape)
						X_test_one = np.hstack([X_test_one, X_test_tmp])
						X_hat = test_model.predict(X_test_one, batch_size)

						X_hat_predict_one = X_hat[0, 1]
						X_hat_predict_one = X_hat_predict_one[np.newaxis, np.newaxis, :, :, :]
						result_list.append(X_hat_predict_one)

					else:
						X_test_one = result_list[-1]
						X_test_tmp = np.zeros(X_test_one.shape)
						X_test_one = np.hstack([X_test_one, X_test_tmp])
						X_hat = test_model.predict(X_test_one, batch_size)

						X_hat_predict_one = X_hat[0, 1]
						X_hat_predict_one = X_hat_predict_one[np.newaxis, np.newaxis, :, :, :]
						result_list.append(X_hat_predict_one)
		
		else:
			for idx in range(0, len(key_frame_check) - 1):
				for predict_idx in range(key_frame_check[idx], key_frame_check[idx+1]):
					# predictする画像がキーフレームの場合
					if predict_idx == key_frame_check[idx]:
						X_test_one = X_test_pad[0, idx]
						X_test_one = X_test_one[np.newaxis, np.newaxis, :, :, :]
						X_test_tmp = np.zeros(X_test_one.shape)
						X_test_one = np.hstack([X_test_one, X_test_tmp])
						X_hat = test_model.predict(X_test_one, batch_size)

						X_hat_predict_one = X_test_pad[0, idx]
						X_hat_predict_one = X_hat_predict_one[np.newaxis, np.newaxis, :, :, :]
						result_list.append(X_hat_predict_one)
					# predictする画像がキーフレームの次の画像の場合(キーフレームをもとにpredictする場合)
					elif predict_idx == key_frame_check[idx] + 1:
						X_test_one = X_test_pad[0, idx]
						X_test_one = X_test_one[np.newaxis, np.newaxis, :, :, :]
						X_test_tmp = np.zeros(X_test_one.shape)
						X_test_one = np.hstack([X_test_one, X_test_tmp])
						X_hat = test_model.predict(X_test_one, batch_size)

						X_hat_predict_one = X_hat[0, 1]
						X_hat_predict_one = X_hat_predict_one[np.newaxis, np.newaxis, :, :, :]
						result_list.append(X_hat_predict_one)
					# predictすをもとにpredictする場合
					else:
						X_test_one = result_list[-1]
						X_test_tmp = np.zeros(X_test_one.shape)
						X_test_one = np.hstack([X_test_one, X_test_tmp])
						X_hat = test_model.predict(X_test_one, batch_size)

						X_hat_predict_one = X_hat[0, 1]
						X_hat_predict_one = X_hat_predict_one[np.newaxis, np.newaxis, :, :, :]
						result_list.append(X_hat_predict_one)

		# 推論結果を一つにまとめる
		X_hat_flat = result_list[0]
		for X_hat_np in result_list[1:]:
			X_hat_flat = np.hstack([X_hat_flat, X_hat_np])

		X_hat_flat[0, 0] = X_test_pad[0, 0]

		# 推論結果からパディングを外す
		X_hat_no_pad = X_hat_flat[:,:, :X_test.shape[2], :X_test.shape[3]]

		# 画像の出力
		decompress = X_hat_no_pad * 255
		decompress = decompress - difference_first

		decompress = np.where(decompress > 255, 255, decompress)
		decompress = np.where(decompress < 0, 0, decompress)

		count = 0 + start_image_index

		for i in range(X_test_shape[0]):
			for j in range(start_image_index, end_image_index+1):
				tmp_np = decompress[i,j-start_image_index,:,:,:]
				tmp_np = tmp_np.astype('uint8')
				tmp_image = Image.fromarray(tmp_np)
				tmp_image.save(os.path.join(OUTPUT_DIR, file_names[count]))
				count += 1

