#Python 2.7.12

import numpy as np
#softmax function for x , x is numpy matrix with size of [nx1]
def softmax(x): 
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

#sigmoid function at x (element wise), x is numpy matrix 
def sigmoid(x):
    ### YOUR CODE HERE:
    x = np.clip(x, -500, 500)
    return 1.0 / (1 + np.exp(-x))

#derivation of sigmoid function at x
def dsigmoid(x):	
	tmp=sigmoid(x)
	return tmp-np.power(tmp,2.0)

#Rectified Linear Unit at x
def relu(x):
	return (x+abs(x))/2.0
#derivation of Rectified Linear Unit at x
def drelu(x):
	return np.matrix(x>0)*1.0

def identity(x):
	return x
def didentity(x):
	return 1.0


##################### test case #####################
def test_softmax():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests on softmax...")
    for inp,out in (
        ([1,2],[0.26894142,0.73105858]),
        ([1001,1002],[0.26894142,0.73105858]),
        ([-1001,-1002],[0.73105858, 0.26894142])
        ):
        inp_matrix=np.matrix(inp).reshape((2,1))
        test_matrix=np.matrix(out).reshape((2,1))
        ans=softmax(inp_matrix)
        assert np.allclose(test_matrix, ans, rtol=1e-05, atol=1e-06)
def test_sigmoid():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests on sigmoid...")
    for inp,out in ( 
        ([1,2],[0.73105858, 0.88079708]),
        ([-1,-2],[0.26894142, 0.11920292]),
        ([1001,-1001],[1.0,0.0]),
        ):
        inp_matrix=np.matrix(inp).reshape((2,1))
        test_matrix=np.matrix(out).reshape((2,1))
        ans=sigmoid(inp_matrix)
        assert np.allclose(test_matrix, ans, rtol=1e-05, atol=1e-06)

if __name__ == "__main__":
    test_softmax()
    test_sigmoid()