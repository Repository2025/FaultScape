import os
import sys
import torch
import time
import pickle
import random
from model.model_semantic_spec_mutation import MLP




def get_batch(x_pos, x_neg, idx, bs): 
	x_pos_batch = x_pos[idx: idx+bs]
	x_neg_batch = x_neg[idx: idx+bs]
	return torch.FloatTensor(x_pos_batch), torch.FloatTensor(x_neg_batch)


def get_x_batch(x, idx, bs):  
	x_batch = x[idx: idx+bs]
	return torch.FloatTensor(x_batch)


def load_from_file(file_path):  
	with open(file_path, "rb") as file:
		return pickle.load(file)


def print_parameter_statistics(model):  
	total_num = [p.numel() for p in model.parameters()]
	trainable_num = [p.numel() for p in model.parameters() if p.requires_grad]
	print("Total parameters: {}".format(sum(total_num)))  #  144
	print("Trainable parameters: {}".format(sum(trainable_num)))  # 144


if __name__ == "__main__":
	task_id = sys.argv[1]
	# task_id = 1
	root = "./FaultScape/LLM_FL_Result/FL/ranking_task/run_model/data/{}/".format(task_id)
	
	train_x_pos = load_from_file(root + 'train/x_pos.pkl')
	train_x_neg = load_from_file(root + 'train/x_neg.pkl')
	val_x_pos = load_from_file(root + 'val/x_pos.pkl')
	val_x_neg = load_from_file(root + 'val/x_neg.pkl')
	test_x_data = load_from_file(root + 'test/x.pkl')
	test_y_data = load_from_file(root + 'test/y.pkl')

	
	EPOCHS = 30
	BATCH_SIZE = 64
	USE_GPU = True

	model = MLP()

	if USE_GPU:
		model.cuda()

	parameters = model.parameters()
	optimizer = torch.optim.Adam(parameters, lr=0.001, weight_decay=1e-4)
	print_parameter_statistics(model)

	train_loss_ = []
	val_loss_ = []

	best_result = 1e3   
	print('Start training...')
	for epoch in range(EPOCHS):
		start_time = time.time()

		#       shuffle data before training for each epoch
		random.seed(888)
		random.shuffle(train_x_pos)
		random.seed(888)
		random.shuffle(train_x_neg)


		#           training phase
		model.train()
		total_loss = 0.0
		total = 0

		i = 0
		while i < len(train_x_pos):
			batch = get_batch(train_x_pos, train_x_neg, i, BATCH_SIZE)
			i += BATCH_SIZE
			batch_x_pos, batch_x_neg = batch
			if USE_GPU:
				batch_x_pos, batch_x_neg = batch_x_pos.cuda(), batch_x_neg.cuda()



			model.zero_grad()
			output_pos = model(batch_x_pos)
			output_neg = model(batch_x_neg)

			output_pos = torch.softmax(output_pos, dim=-1)[:, 0]   
			output_neg = torch.softmax(output_neg, dim=-1)[:, 0]

			
			#  hinge loss  
			loss = torch.max(torch.zeros_like(output_pos), 0.15 - (output_pos - output_neg)).mean()  
   


			loss.backward()
			optimizer.step() 

			total += len(batch_x_pos)
			total_loss += loss.item() * len(batch_x_pos)

		train_loss_.append(total_loss / total)

		model.eval()
		total_loss = 0.0
		total = 0

		i = 0
		while i < len(val_x_pos):
			batch = get_batch(val_x_pos, val_x_neg, i, BATCH_SIZE)
			i += BATCH_SIZE
			batch_x_pos, batch_x_neg = batch
			if USE_GPU:
				batch_x_pos, batch_x_neg = batch_x_pos.cuda(), batch_x_neg.cuda()
			output_pos = model(batch_x_pos)
			output_neg = model(batch_x_neg)
			output_pos = torch.softmax(output_pos, dim=-1)[:, 0]
			output_neg = torch.softmax(output_neg, dim=-1)[:, 0]


			
			labels_pos = torch.ones_like(output_pos)  
			labels_neg = torch.zeros_like(output_neg)
			loss_pos = torch.nn.functional.binary_cross_entropy(output_pos, labels_pos) 
			loss_neg = torch.nn.functional.binary_cross_entropy(output_neg, labels_neg) 
			loss = ((loss_pos + loss_neg).mean())/2



			total += len(batch_x_pos)
			total_loss += loss.item() * len(batch_x_pos)

		val_loss_.append(total_loss / total)

		if val_loss_[-1] < best_result:
			torch.save(model.state_dict(), "./model_save/model_params_{}.pkl".format(task_id))
			torch.save(model, "./model_save/model_{}.pkl".format(task_id))
			best_result = val_loss_[-1]
			print("Saving model: epoch_{} ".format(epoch + 1) + "=" * 20)

		end_time = time.time()
		print('[Epoch: %3d/%3d]\nTraining Loss: %.10f,\t\tValidation Loss: %.10f\nTime Cost: %.3f s'
			  % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch], end_time - start_time))
	


	

	model.load_state_dict(torch.load("./model_save/model_params_{}.pkl".format(task_id)))
	model.eval()

	with open("./FaultScape/LLM_FL_Result/FL/d4j_data/statements.pkl", "rb") as file:
		statements = pickle.load(file)
		print("statements: ",len(statements.keys()),len(statements['Chart_1']) )



	with open("../faulty_statement_set.pkl", "rb") as file:
		faulty_statements = pickle.load(file)
		print("faulty_statements_set:",(faulty_statements) )




	with open("./result/{}.txt".format(task_id), "w") as result_file:
		for version in test_x_data:
			result_file.write("==={}\n".format(version))
			version_x_data = test_x_data[version]  
			# print(version_x_data)
			predict_score = torch.empty(0)
			if USE_GPU:
				predict_score = predict_score.cuda()

			i = 0
			while i < len(version_x_data):
				test_inputs = get_x_batch(version_x_data, i, BATCH_SIZE)
				i += BATCH_SIZE
				if USE_GPU:
					test_inputs = test_inputs.cuda()

				output = model(test_inputs)   
				output_softmax = torch.softmax(output, dim=-1)[:, 0]
				predict_score = torch.cat((predict_score, output_softmax)) 

			predict_score = predict_score.cpu().detach().numpy().tolist()  

			sus_lines = statements[version]  



			
			sus_pos_rerank_dict = {}
			for i, line in enumerate(sus_lines):
				sus_pos_rerank_dict[line] = predict_score[i] 

			sorted_sus_list = sorted(sus_pos_rerank_dict.items(), key=lambda x: x[1], reverse=True)



			
			out_suspicious_dir = "./sus_pos_rerank_new/{}/".format(version)
			if not os.path.exists(out_suspicious_dir):
				os.makedirs(out_suspicious_dir)

			with open(os.path.join(out_suspicious_dir, "ranking.txt"), "w") as file:
				for (line, score) in sorted_sus_list:
					file.write("{} {}\n".format(line, score))





			rerank_sus_lines = [line for (line, score) in sorted_sus_list]  
			rerank_sus_scores = [float(score) for (line, score) in sorted_sus_list] 

		
			current_faulty_statements = faulty_statements[version]




			for one_position_set in current_faulty_statements:  
				current_min_index = 1e8
				for buggy_line in one_position_set: 
					if buggy_line not in rerank_sus_lines: 
						continue
					buggy_index = len(rerank_sus_scores) - rerank_sus_scores[::-1].index(rerank_sus_scores[rerank_sus_lines.index(buggy_line)])
					if buggy_index < current_min_index:
						current_min_index = buggy_index
				if current_min_index == 1e8:
					continue


				result_file.write(str(current_min_index) + "\n")
