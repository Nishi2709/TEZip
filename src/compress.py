import os
import glob
import numpy as np

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from data_utils import SequenceGenerator, data_padding

from scipy import sparse
from scipy.sparse import *

from numba import cuda

from PIL import Image, UnidentifiedImageError
import zstd

import time

def error_bound(origine, diff, mode, value, GPU_FLAG, xp):
	if value[0] == 0: return diff  # Do nothing if lossless compression
	Df = diff.reshape(diff.shape[0]*diff.shape[2], diff.shape[3]*diff.shape[4], diff.shape[5])
	Bf = origine.reshape(origine.shape[0]*origine.shape[2], origine.shape[3]*origine.shape[4], origine.shape[5])

	if mode == "abs":
		E = xp.abs(value[0])
	elif mode == "rel":
		diff_max = Bf.max(axis=(1), keepdims=True)
		diff_min = Bf.min(axis=(1), keepdims=True)
		E = (diff_max - diff_min) * value[0]
	elif mode == "absrel":
		if value[1] == 0:
			return diff
		diff_max = Bf.max(axis=(1), keepdims=True)
		diff_min = Bf.min(axis=(1), keepdims=True)
		abs_value = xp.abs(value[0])
		rel_value = (diff_max - diff_min) * value[1]
		E = xp.where(abs_value < rel_value, abs_value, rel_value)
	elif mode == "pwrel":
		E = origine * value[0]  # Error abs
	
	Du = Df + E  # Du: Upper error bound
	Dl = Df - E  # Dl: Lower error bound
	
	u_per_channel_array = np.full((Df.shape[0],Df.shape[2]),float(np.inf)) # Temp upper error bound
	l_per_channel_array = np.full((Df.shape[0],Df.shape[2]),-float(np.inf)) # Temp lower error bound

	if GPU_FLAG:
		Df = xp.asnumpy(Df)
		Du = xp.asnumpy(Du)
		Dl = xp.asnumpy(Dl)
		u_per_channel_array = xp.asnumpy(u_per_channel_array)
		l_per_channel_array = xp.asnumpy(l_per_channel_array)

	head_per_channel_array = np.zeros((Df.shape[0],Df.shape[2]), dtype=int)

	for i in range(Df.shape[1]):
		# channelごとの許容範囲に収まらないタイミングのbool
		Du_target = Du[:,i,:]
		Dl_target = Dl[:,i,:]
		boundary_bool = np.minimum(u_per_channel_array, Du_target) - np.maximum(l_per_channel_array, Dl_target) < 0
		# 許容範囲を超えたチャネルのインデックスを取得
		indices = np.where(boundary_bool)
		# インデックスをタプルのリストに変換
		index_array = list(zip(indices[0], indices[1]))

		if len(index_array) >= 1:
			for img_index, channel_index in index_array:
				Df[img_index, head_per_channel_array[img_index, channel_index]:i, channel_index] = (u_per_channel_array[img_index, channel_index] + l_per_channel_array[img_index, channel_index])/2 # compute a median [l, u]
				u_per_channel_array[img_index, channel_index] = float(np.inf) # reinit
				l_per_channel_array[img_index, channel_index] = -u_per_channel_array[img_index, channel_index] # reinit
				head_per_channel_array[img_index, channel_index] = i # update to the fist index of the next product(intersect) set

		u_per_channel_array = np.where(u_per_channel_array > Du_target, Du_target, u_per_channel_array) # accumulate product(intersect) set
		l_per_channel_array = np.where(l_per_channel_array < Dl_target, Dl_target, l_per_channel_array) # accumulate product(intersect) set	

	# 画像のループ
	for img_index in range(Df.shape[0]):
		# チャンネルのループ
		for channel_index in range(Df.shape[2]):
			Df[img_index, head_per_channel_array[img_index, channel_index]:Df.shape[1], channel_index] = (u_per_channel_array[img_index, channel_index] + l_per_channel_array[img_index, channel_index])/2 # compute the last median [l, u]

	if GPU_FLAG:
		Df = xp.asarray(Df)

	return Df.reshape(diff.shape) # convert back to 2D array

def finding_difference(arr):
    arr_f = arr.flatten()
    arr_f[1:] = arr_f[:-1] - arr_f[1:]

    return arr_f.reshape(arr.shape)


def takeSecond(elem):
	return elem[1]


def replacing_based_on_frequency(arr, table, xp):
	result = arr.copy()

	for idx, num in enumerate(table):
		result = xp.where(result == num, idx, result)

	return result

def run(WEIGHTS_DIR, DATA_DIR, OUTPUT_DIR, PREPROCESS, WINDOW_SIZE, THRESHOLD, MODE, BOUND_VALUE, GPU_FLAG, VERBOSE, ENTROPY_RUN):

	if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

	file_paths = sorted(glob.glob(os.path.join(DATA_DIR, '*')))

	if len(file_paths) == 0:
		print("ERROR:", DATA_DIR, "is an empty or non-existent directory")
		exit()

	try:
		origine_img = np.array(Image.open(file_paths[0]))

		image_mode = Image.open(file_paths[0]).mode
		# if image is other than RGB and grayscale, exit()
		if all([image_mode != 'RGB', image_mode != 'L']):
			print("ERROR: input image is {0}. Only RGB and grayscale are supported.".format(image_mode))
			exit()

		# gray scale convert RGB
		isRGB = image_mode == 'RGB' # Identify input image channel.
		origine_img = np.array(Image.open(file_paths[0])) if isRGB else np.array(Image.open(file_paths[0]).convert('RGB'))

		origine_img = origine_img[np.newaxis, np.newaxis, :, :, :]
		files = [os.path.basename(file_paths[0])]
		for path in file_paths[1:]:
			img = np.array((Image.open(path))) if isRGB else np.array((Image.open(path).convert('RGB')))
			img = img[np.newaxis, np.newaxis, :, :, :]
			origine_img = np.hstack([origine_img, img])
			files.append(os.path.basename(path))
	except PermissionError as e:
		print(DATA_DIR, "contains files or folders that are not images.")
		exit()
	except IndexError as e:
		print(DATA_DIR, "contains files or folders that are not images.")
		exit()
	except UnidentifiedImageError as e:
		print(DATA_DIR, "contains files or folders that are not images.")
		exit()

	with open(os.path.join(OUTPUT_DIR, 'filename.txt'), 'w', encoding='UTF-8') as f:
		f.write(f"{int(isRGB)}\n") # Append rgb status to filename for later possible grayscale recovery.
		for file_name in files:
			f.write("%s\n" % file_name)

	X_test = origine_img.astype(np.float32) /255

	batch_size = 10
	nt = X_test.shape[1] # 画像の枚数

	weights_file = os.path.join(WEIGHTS_DIR, 'prednet_weights.hdf5')
	json_file = os.path.join(WEIGHTS_DIR, 'prednet_model.json')

	# Load trained model

	start_time = time.time()

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

	# モデルセッティング
	test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
	input_shape = list(train_model.layers[0].batch_input_shape[2:])
	input_shape.insert(0, None)
	inputs = Input(shape=tuple(input_shape))
	predictions = test_prednet(inputs)
	test_model = Model(inputs=inputs, outputs=predictions)

	end_time = time.time()
	elapsed_time = end_time - start_time
	if VERBOSE: print ("load_trained_model:{0}".format(elapsed_time) + "[sec]")

	# 推論用に元画像にパディング
	X_test_pad = data_padding(X_test)

	if test_model.input.shape[2] != X_test_pad.shape[2] or test_model.input.shape[3] != X_test_pad.shape[3]:
		print("ERROR:Image size is out of scope for this model.")
		print("Compatible sizes for this model are height", test_model.input.shape[2] - 7, "to", test_model.input.shape[2], "and width",  test_model.input.shape[3] - 7, "to", test_model.input.shape[3])
		exit()

	key_frame = np.zeros(origine_img.shape, dtype='uint8')

	origine_list = []
	predict_list = []

	# warm up

	start_time = time.time()

	for w_idx in range(PREPROCESS):
		key_frame[0, w_idx] =  origine_img[0, w_idx]
		X_test_one = X_test_pad[0, w_idx]
		X_test_one = X_test_one[np.newaxis, np.newaxis, :, :, :]
		X_test_tmp = np.zeros(X_test_one.shape)
		X_test_one = np.hstack([X_test_one, X_test_tmp])
		X_hat = test_model.predict(X_test_one, batch_size)

		warm_up_frame = X_hat[0, 0]
		warm_up_frame = warm_up_frame[np.newaxis, np.newaxis, :, :, :]
		if w_idx == 0:
			predict_stack_np = warm_up_frame
		else:
			predict_stack_np = np.hstack([predict_stack_np, warm_up_frame])

	if PREPROCESS != 0:
		origine_list.append(origine_img[:, :PREPROCESS])
		predict_list.append(predict_stack_np)
		predict_stack_np = X_hat[0, 0]
		predict_stack_np = predict_stack_np[np.newaxis, np.newaxis, :, :, :]

		origine_stack_np = origine_img[0, PREPROCESS]
		origine_stack_np = origine_stack_np[np.newaxis, np.newaxis, :, :, :]

	end_time = time.time()
	elapsed_time = end_time - start_time
	if VERBOSE: print ("warm_up:{0}".format(elapsed_time) + "[sec]")

	# predict
	
	start_time = time.time()

	key_idx = PREPROCESS + 1
	stop_point = 0
	idx = PREPROCESS + 1
	while(idx < X_test_pad.shape[1]):
		if idx == key_idx:
			X_test_one = X_test_pad[0, idx - 1]
			key_frame[0, idx - 1] =  origine_img[0, idx - 1]
		else:
			X_test_one = predict_stack_np[0, -1]

		X_test_one = X_test_one[np.newaxis, np.newaxis, :, :, :]
		X_test_tmp = np.zeros(X_test_one.shape)
		X_test_one = np.hstack([X_test_one, X_test_tmp])
		X_hat = test_model.predict(X_test_one, batch_size)

		X_hat_predict_one = X_hat[0, 1]
		X_hat_predict_one = X_hat_predict_one[np.newaxis, np.newaxis, :, :, :]

		X_test_origine_one = origine_img[0, idx]
		X_test_origine_one = X_test_origine_one[np.newaxis, np.newaxis, :, :, :]

		if idx == 1:
			predict_stack_np = X_hat[0, 0]
			predict_stack_np = predict_stack_np[np.newaxis, np.newaxis, :, :, :]
			predict_stack_np = np.hstack([predict_stack_np, X_hat_predict_one])
			origine_stack_np = origine_img[0, :2]
			origine_stack_np = origine_stack_np[np.newaxis, :, :, :]
		else:
			predict_stack_np = np.hstack([predict_stack_np, X_hat_predict_one])
			origine_stack_np = np.hstack([origine_stack_np, X_test_origine_one])

		if idx >= key_idx:
			stop_point = np.mean( (X_test_pad[:, key_idx:idx+1] - predict_stack_np[:, 1:])**2 )
			if VERBOSE: print("MSE:", stop_point)

		if (THRESHOLD != None and stop_point > THRESHOLD) or (WINDOW_SIZE != None and (idx - PREPROCESS) % WINDOW_SIZE == 0):
			if VERBOSE: print("move key point")
			origine_result = origine_stack_np[:, :-1]
			origine_list.append(origine_result)
			predict_result = predict_stack_np[:, :-1]
			predict_list.append(predict_result)

			origine_stack_np = origine_img[0, idx]
			origine_stack_np = origine_stack_np[np.newaxis, np.newaxis, :, :, :]
			predict_stack_np = X_hat[0, 0]
			predict_stack_np = predict_stack_np[np.newaxis, np.newaxis, :, :, :]
			if idx == X_test.shape[1] - 1:
				key_frame[0, idx] =  origine_img[0, idx]
				predict_stack_np[0, 0] = X_hat[0, 1]
			key_idx = idx + 1
			stop_point = 0

		idx += 1
	origine_list.append(origine_stack_np)
	predict_list.append(predict_stack_np)

	end_time = time.time()
	elapsed_time = end_time - start_time
	if VERBOSE: print ("predict:{0}".format(elapsed_time) + "[sec]")

	# キーフレームの出力
	key_frame = key_frame.flatten()
	key_frame = key_frame.astype('uint8')
	key_frame_str = key_frame.tostring()

	# zstdでキーフレームを圧縮・出力
	data=zstd.compress(key_frame_str, 9)
	with open(os.path.join(OUTPUT_DIR, "key_frame.dat"), mode='wb') as f:
		f.write(data)

	# GPU無:numpy GPU有:cupyに設定
	if GPU_FLAG:
		# tensorflowが占有しているメモリを解放
		cuda.select_device(0)
		cuda.close()
		import cupy as xp
	else:
		import numpy as xp

	error_bound_time = 0

	# エラーバウンド機構実施の準備

	start_time = time.time()

	# shapeが変わるインデックスを記録するリスト
	shape_change_indices_li = []
	for i in range(len(origine_list)):
		if i == 0:
			shape_change_indices_li.append(i)
		else:
			# 前の要素と形状を比較
			if origine_list[i].shape != origine_list[i-1].shape:
				shape_change_indices_li.append(i)

	difference_list = []
	for i in range(len(shape_change_indices_li)):
		if i != len(shape_change_indices_li)-1:
			origine_array = np.array(origine_list[shape_change_indices_li[i]:shape_change_indices_li[i+1]])
			predict_array = np.array(predict_list[shape_change_indices_li[i]:shape_change_indices_li[i+1]])
		else:
			origine_array = np.array(origine_list[shape_change_indices_li[i]:])
			predict_array = np.array(predict_list[shape_change_indices_li[i]:])

		origine_array = np.divide(origine_array, 255.000)

		# 推論画像からパディングを外す
		predict_array_no_pad = predict_array[:, :, :, :X_test.shape[2], :X_test.shape[3]]

		# GPU無:numpy GPU有:cupyに変換
		if GPU_FLAG:
			origine_array = xp.asarray(origine_array)
			predict_array_no_pad = xp.asarray(predict_array_no_pad)
			X_hat_1=xp.multiply(predict_array_no_pad[:,:],255.000,casting='unsafe')
			X_test_1=xp.multiply(origine_array[:,:],255.000,casting='unsafe')
		else:
			X_hat_1=np.multiply(predict_array_no_pad[:,:],255.000,casting='unsafe')
			X_test_1=np.multiply(origine_array[:,:],255.000,casting='unsafe')

		X_test_1 = X_test_1.astype(int)
		X_hat_1 = X_hat_1.astype(int)

		difference = (X_hat_1[:, :] - X_test_1[:, :])
		difference[:, :, 0] = 0
		if not (PREPROCESS != 0 and i == 0):
			difference = error_bound(X_test_1, difference, MODE, BOUND_VALUE, GPU_FLAG, xp)
		
		difference_list.append(difference)

	end_time = time.time()
	error_bound_time = end_time - start_time

	if VERBOSE: print ("error_bound:{0}".format(error_bound_time) + "[sec]")

	# 推論結果をまとめる　GPU&pwrelの場合はこの段階でcupyに切り替わる
	difference_model = difference_list[0][0,:]
	for X_hat_np in difference_list[1:]:
		for i in range(X_hat_np.shape[0]):
			difference_model = xp.hstack([difference_model, X_hat_np[i]])

	difference_model = difference_model.astype('int16')

	# Density-based Spatial Encoding

	start = time.time()

	difference_model = finding_difference(difference_model)
	difference_model=difference_model.flatten()

	elapsed_time = time.time() - start

	if VERBOSE: print ("finding_difference:{0}".format(elapsed_time) + "[sec]")

	if ENTROPY_RUN:
		# エントロピー符号化のテーブル作成のために適当な正の整数に変換(1600との差分として保存)
		difference_model = xp.subtract(1600, difference_model)

		# エントロピー符号化用のテーブル作成
		start = time.time()
		table = []
		x_elem = difference_model.flatten()
		y_elem = xp.bincount(x_elem)
		ii_elem = xp.nonzero(y_elem)[0]
		d = list(zip(ii_elem, y_elem[ii_elem]))
		d.sort(key=takeSecond, reverse=True)
		for key, value in d :
			table.append(key)

		table_xp = xp.array(table, dtype='int16')

		elapsed_time = time.time() - start

		if VERBOSE: print ("table_create:{0}".format(elapsed_time) + "[sec]")

		start = time.time()
		# エントロピー符号化
		difference_model = replacing_based_on_frequency(difference_model, table_xp, xp)

		elapsed_time = time.time() - start

		if VERBOSE: print ("replacing_based_on_frequency:{0}".format(elapsed_time) + "[sec]")

	result_difference = difference_model.flatten()

	# cupyに変換していたらnumpyに戻す(他ライブラリが絡む&append未実装のバージョンがあるため)
	if GPU_FLAG:
		result_difference = xp.asnumpy(result_difference)

	if ENTROPY_RUN:
		# 差分配列の末尾にエントロピー符号化のテーブルを仕込んでおく
		s_np = np.array(table, dtype='int16')
		result_difference = np.append(result_difference, s_np)
		result_difference = np.append(result_difference, len(table))
	else:
		result_difference = np.append(result_difference, -1)

	# 差分配列の末尾にshapeとPREPROCESSを仕込んで保存しておく
	for shapes in X_test.shape:
		result_difference = np.append(result_difference, shapes)
	result_difference = np.append(result_difference, PREPROCESS)

	result_difference = result_difference.astype(np.int16)
	result_difference_str = result_difference.tostring()

	# zstdで差分を圧縮・出力
	data=zstd.compress(result_difference_str, 9)
	with open(os.path.join(OUTPUT_DIR, "entropy.dat"), mode='wb') as f:
		f.write(data)

