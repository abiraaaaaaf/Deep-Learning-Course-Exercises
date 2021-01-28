#Python 2.7.12

import numpy as np
import activation_functions
import error_functions

################## some map for converting string to function ##################
################################# see usage below ##############################
diff_activation_function_map={}
activation_function_map={}
loss_function_map={}

activation_function_map["softmax"]		=activation_functions.softmax
activation_function_map["sigmoid"]		=activation_functions.sigmoid
diff_activation_function_map["sigmoid"]	=activation_functions.dsigmoid
activation_function_map["relu"]			=activation_functions.relu
diff_activation_function_map["relu"]	=activation_functions.drelu
activation_function_map["identity"]		=activation_functions.identity
diff_activation_function_map["identity"]=activation_functions.didentity
softmax=activation_function_map["softmax"]

loss_function_map["softmax_ce"]=error_functions.softmax_ce_derivation

def NN_output(config,w_list,b_list,x):
	last_out=x
	for i in range(len(config)):
		act=activation_function_map[config[i]["act_name"]]
		last_out=act(w_list[i]*last_out+b_list[i])
	return last_out

def NN_softmax_output_and_error(config,w_list,b_list,x):
	global softmax
	return softmax(NN_output(config,w_list,b_list,x))

def compute_gradient(config,w_list,b_list,loss_function_name,x,y):

	"""
	* config[i] : config of i th layer 
		config[i]["num"] : number of neuron in i th layer
		config[i]["act_name"] : activation function of i th layer
			pissible values:
				"sigmoid"
				"relu"
				"identity"
	* w_list[i] : current weights matrix that make i th layer
	* b_list[i] : current weights matrix  that make i th layer

	* loss_function_name : name of loss_function as string
		possible values:
			"softmax_ce"
	
	* x : input
	* y : target output
	"""
	
	z=[]
	o=[x]
	
	#forward:
	for i in range(len(config)):
		act=activation_function_map[config[i]["act_name"]]
		# if i==0:
		# 	#z[0].append([x])
		# 	for l in x:
		# 		z.append(l)
		# 	#z = np.array([x])
		z.append(w_list[i]*o[-1]+b_list[i])
		# else:
		# 	act = activation_function_map[config[i]["act_name"]]  # you can use "act(input)"
		# 	z.append(act(w_list[i-1]*z[i-1]+b_list[i-1]))
		o.append(act(z[-1]))
	
	#backward:
	loss_function=loss_function_map[loss_function_name]
	dE_dlastO=loss_function(o[-1],y)
	dE_dw=[]
	dE_db=[]
	# dE_dw[len(config)]=dE_dlastO
	# dE_db[len(config)]=dE_dlastO
	for i in range(len(config)-1,-1,-1):

		d_act = diff_activation_function_map[config[i]["act_name"]]
		# dE_dw[i].append(dE_dw[i+1]*d_act(z[i]))
		# dE_db[i].append(dE_db[i+1]*d_act(1))
		dE_dlastZ = np.multiply(dE_dlastO , np.transpose(d_act(z[i])))
		dE_dlastO = dE_dlastZ * w_list[i] * 2
		##
		dE_db = [np.transpose(dE_dlastZ)] + dE_db
		dE_dw = [np.transpose(o[i]*dE_dlastZ) ] + dE_dw


	return o[-1],dE_dw,dE_db

