#Python 2.7.12

import numpy as np
import etc
import gradient

feture_number=20
class_number=3
batch_size = 50
learning_rate=0.1
training_epoch = 100

########################## load dataset ##########################

dataset_x_test,dataset_y_test,dataset_x_tarin,dataset_y_train=etc.load_dataset()
# dataset_x_tarin & dataset_x_test : list of numpy matrix 20x1
# dataset_y_tarin & dataset_y_test : list of numpy matrix 3x1  (one-hot encoded)

############################ make NN ############################
config=[]
w_list=[]
b_list=[]
#first layer

ln1=10
config.append({"num":ln1,"act_name":"sigmoid"})
w_list.append(np.matrix(np.random.normal(size=(ln1,feture_number),scale=0.2)).astype("double"))
b_list.append(np.matrix(np.zeros((ln1,1))).astype("double"))
#second layer
ln2=class_number
config.append({"num":ln2,"act_name":"identity"})
w_list.append(np.matrix(np.random.normal(size=(ln2,config[-2]["num"]),scale=0.2)).astype("double"))
b_list.append(np.matrix(np.zeros((ln2,1))).astype("double"))


accuracy_rate,ce_error=etc.accuracy(config,w_list,b_list,dataset_x_tarin,dataset_y_train)

############################ make NN ############################

for epoch in range(training_epoch):
	matrisw = []
	matrisb = []
	dataset_x_tarin,dataset_y_train=etc.shuffle_pair(dataset_x_tarin,dataset_y_train)
	for i in range(len(b_list)):
		matrisw.append(np.matrix(np.zeros(w_list[i].shape)))
		matrisb.append(np.matrix(np.zeros(b_list[i].shape)))
		# matrisb.append(np.matrix(np.zeros(len(w_list[i]))))
		#matrisb.append(np.matrix(np.zeros(len(b_list[i]))))
	for i in range(len(dataset_x_tarin)):
		y,dw,db=gradient.compute_gradient(config,w_list,b_list,"softmax_ce",dataset_x_tarin[i],dataset_y_train[i])
		for k in range(len(config)):
			matrisw[k] = matrisw[k] + dw[k]
			matrisb[k] = matrisb[k] + db[k]
		if i % batch_size == 0:
			for j in range(len(config)):
				w_list[j]=w_list[j]-learning_rate*matrisw[j]
				b_list[j]=b_list[j]-learning_rate*matrisb[j]
				matrisw = []
				matrisb = []
				for i in range(len(b_list)):
					#########
					matrisw.append(np.matrix(np.zeros(w_list[i].shape)))
					matrisb.append(np.matrix(np.zeros(b_list[i].shape)))

print(etc.accuracy(config,w_list,b_list,dataset_x_tarin,dataset_y_train),etc.accuracy(config,w_list,b_list,dataset_x_test,dataset_y_test))
	


