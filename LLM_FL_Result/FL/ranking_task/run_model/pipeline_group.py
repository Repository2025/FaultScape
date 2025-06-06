import os
import random
import pickle
import sys

train_val_split = 0.9


with open("./FaultScape/LLM_FL_Result/FL/ranking_task/run_model/x.pkl", "rb") as file:
	x_data = pickle.load(file)
with open("./FaultScape/LLM_FL_Result/FL/ranking_task/run_model/y.pkl", "rb") as file:
	y_data = pickle.load(file)

with open("./FaultScape/LLM_FL_Result/FL/d4j_data/src_code.pkl", "rb") as file1:
	src_code = pickle.load(file1)


# 10-fold cross-validation
step = 40
versions = list(x_data.keys())


total_code = 0
for key in x_data.keys():
	total_code = total_code + len(x_data[key])



random.seed(888)
random.shuffle(versions)

for group_index in range(10):
	train_pos_x = []
	train_neg_x = []
	
	data_path = "./FaultScape/LLM_FL_Result/FL/ranking_task/run_model/data/{}/".format(group_index + 1)
	test_versions = versions[group_index * step: (group_index + 1) * step]  
	train_versions = [version for version in versions if version not in test_versions] 



	total_pos = 0
	for version in train_versions:
		
		pos_indices = [index for (index, value) in enumerate(y_data[version]) if value == 0]
		neg_indices = [index for (index, value) in enumerate(y_data[version]) if value == 1]
		assert len(pos_indices) + len(neg_indices) == len(y_data[version])
		
		total_pos = total_pos + len(pos_indices)


		for pos_index in pos_indices:  
			neg_indices = neg_indices
			random.seed(888)
			random.shuffle(neg_indices)

			for neg_index in neg_indices[:10]:  		
				train_pos_x.append( x_data[version][pos_index]) 
				train_neg_x.append( x_data[version][neg_index])
	
	test_x = {}
	test_y = {}
	for version in test_versions:
		test_x[version] = x_data[version]
		test_y[version] = y_data[version]
	
	random.seed(888)
	random.shuffle(train_pos_x)
	random.seed(888)
	random.shuffle(train_neg_x)
	
	val_pos_x = train_pos_x[round(len(train_pos_x)*train_val_split):]
	val_neg_x = train_neg_x[round(len(train_neg_x)*train_val_split):]

	train_pos_x = train_pos_x[:round(len(train_pos_x)*train_val_split)]
	train_neg_x = train_neg_x[:round(len(train_neg_x)*train_val_split)]




	if not os.path.exists(data_path):
		os.makedirs(data_path)

	train_dir = os.path.join(data_path, "train/")
	if not os.path.exists(train_dir):
		os.makedirs(train_dir)

	val_dir = os.path.join(data_path, "val/")
	if not os.path.exists(val_dir):
		os.makedirs(val_dir)

	test_dir = os.path.join(data_path, "test/")
	if not os.path.exists(test_dir):
		os.makedirs(test_dir)

	with open(os.path.join(train_dir, "x_pos.pkl"), "wb") as file: 
		pickle.dump(train_pos_x, file)
	with open(os.path.join(train_dir, "x_neg.pkl"), "wb") as file:  
		pickle.dump(train_neg_x, file)

	with open(os.path.join(val_dir, "x_pos.pkl"), "wb") as file:  
		pickle.dump(val_pos_x, file)
	with open(os.path.join(val_dir, "x_neg.pkl"), "wb") as file:
		pickle.dump(val_neg_x, file)
		
	with open(os.path.join(test_dir, "x.pkl"), "wb") as file: 
		pickle.dump(test_x, file)
	with open(os.path.join(test_dir, "y.pkl"), "wb") as file: 
		pickle.dump(test_y, file)

