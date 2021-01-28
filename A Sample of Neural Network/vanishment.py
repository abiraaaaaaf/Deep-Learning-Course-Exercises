#Python 2.7.12

import numpy as np
import gradient

###############################
layer_number = 20
###############################
x=np.matrix(np.random.normal(size=(3,1)))
y=np.matrix([[0.0],[1.0],[0.0]])

for act in ("sigmoid","relu"):
	s=0.0
	for j in range(100):

		config=[]
		w_list=[]
		b_list=[]

		for i in range(layer_number):
			config.append({"num":3,"act_name":act})
			w_list.append(np.matrix(np.random.normal(size=(3,3))).astype("double"))
			b_list.append(np.matrix(np.random.normal(size=(3,1))).astype("double"))

		y,dw,db=gradient.compute_gradient(config,w_list,b_list,"softmax_ce",x,y)
		s+=abs(dw[0]).mean()
	print "[Activation Function = %s\tLayer = %d] : %f" %(act,layer_number,s)