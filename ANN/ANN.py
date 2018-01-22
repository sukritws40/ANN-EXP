import numpy as np
import matplotlib.pyplot as plt

#Learning rate
#L_r = [0.001,0.01,0.1,1,10,100,1000]
L_r = [0.01]
#Number of Hidden layer's neuron
hiddenSize = 5

# Sigmoid Function
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# Derivative of the sigmoid function
def sigmoid_output_to_derivative(output):
    return output*(1-output)
    
X = 40 * np.random.random_sample((10, 1)) - 20
#print(X.shape)
                
y = (X**3)    
#print(y.shape)

for n in L_r:
    print("\nTraining With Learning rate: " + str(n))
    np.random.seed(1)

    # randomly initialize our weights in range of [-5,5]
    W1 = 10*np.random.random((1,hiddenSize)) - 5
    W2 = 10*np.random.random((hiddenSize,1)) - 5

    for j in range(1000):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0,W1))
        layer_2 = sigmoid(np.dot(layer_1,W2))


        layer_2_error = layer_2 - y

        if (j% 1) == 0:
            print("Error after "+str(j+1)+" iterations:" + str(np.mean(np.abs(layer_2_error))))

        layer_2_delta = layer_2_error*sigmoid_output_to_derivative(layer_2)

        layer_1_error = layer_2_delta.dot(W2.T)
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        # Update the weight
        W2 += L_r * (layer_1.T.dot(layer_2_delta))
        W1 += L_r * (layer_0.T.dot(layer_1_delta))
    
print("\ninput -> X")
print(X)

print("\ny")
print(y)
  
print("\nafter training -> Predict")  
print(layer_2)

plot_1 = plt.scatter(X, y)
plot_2 = plt.scatter(X, layer_2, marker='x', color='r')

plt.legend((plot_1, plot_2),
           ('dataset (X)', 'Predicted value'),
           scatterpoints=1,
           loc='lower right',
           ncol=3,
           fontsize=15)


plt.xlabel('X')
plt.ylabel('y')
plt.show()








