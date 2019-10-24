import numpy as np


class NeuralNetwork:

    def __init__(self, lr, batchSize, epochs):
        # Method to initialize a Neural Network Object
        # Parameters
        # lr - learning rate
        # batchSize - Mini batch size
        # epochs - Number of epochs for training
        self.lr = lr
        self.batchSize = batchSize
        self.epochs = epochs
        self.layers = []

    def addLayer(self, layer):
        # Method to add layers to the Neural Network
        self.layers.append(layer)

    def train(self, trainX, trainY, validX=None, validY=None):
        # Method for training the Neural Network
        # Input
        # trainX - A list of training input data to the neural network
        # trainY - Corresponding list of training data labels
        # validX - A list of validation input data to the neural network
        # validY - Corresponding list of validation data labels

        # The methods trains the weights and baises using the training data(trainX, trainY)
        # Feel free to print accuracy at different points using the validate() or computerAccuracy() functions of this class
        ###############################################
        # TASK 2c (Marks 0) - YOUR CODE HERE
        for r in range(self.epochs):
            for iter in range(trainX.shape[0]//self.batchSize):
                trainx=[]
                trainy=[]
                for i in range(self.batchSize):
                    temp=np.random.randint(0,trainX.shape[0])
                    trainx.append(trainX[temp])
                    trainy.append(trainY[temp])
                trainx=np.array(trainx)
                trainy=np.array(trainy)
                activations = [trainx]
                for layer in self.layers:
                    activations.append(layer.forwardpass(np.array(activations[-1])))
                delta = self.crossEntropyDelta(trainy, activations[-1])
                for i in range(len(self.layers)-1, -1, -1):
                    delta = self.layers[i].backwardpass(np.array(activations[i]), delta)
                    self.layers[i].updateWeights(self.lr)
            print(self.computeAccuracy(validY,self.predict(validX)))
        # raise NotImplementedError
        ###############################################

    def crossEntropyLoss(self, Y, predictions):
        # Input
        # Y : Ground truth labels (encoded as 1-hot vectors) | shape = batchSize x number of output labels
        # predictions : Predictions of the model | shape = batchSize x number of output labels
        # Returns the cross-entropy loss between the predictions and the ground truth labels | shape = scalar
        ###############################################
        # TASK 2a (Marks 3) - YOUR CODE HERE
        
        error = -np.nan_to_num(Y*np.log(predictions),neginf=0)
        mean_error = error.sum()
        return mean_error
        # raise NotImplementedError
        ###############################################

    def crossEntropyDelta(self, Y, predictions):
        # Input
        # Y : Ground truth labels (encoded as 1-hot vectors) | shape = batchSize x number of output labels
        # predictions : Predictions of the model | shape = batchSize x number of output labels
        # Returns the derivative of the loss with respect to the last layer outputs, ie dL/dp_i where p_i is the ith
        #		output of the last layer of the network | shape = batchSize x number of output labels
        ###############################################
        # TASK 2b (Marks 3) - YOUR CODE HERE
        temp=-np.nan_to_num(np.divide(Y, predictions),nan=0,posinf=1)
        return temp
        # raise NotImplementedErrort6
        ###############################################

    def computeAccuracy(self, Y, predictions):
        # Returns the accuracy given the true labels Y and final output of the model
        correct = 0
        for i in range(len(Y)):
            if np.argmax(Y[i]) == np.argmax(predictions[i]):
                correct += 1
        accuracy = (float(correct) / len(Y)) * 100
        return accuracy

    def validate(self, validX, validY):
        # Input
        # validX : Validation Input Data
        # validY : Validation Labels
        # Returns the predictions and validation accuracy evaluated over the current neural network model
        valActivations = self.predict(validX)
        pred = np.argmax(valActivations, axis=1)
        if validY is not None:
            valAcc = self.computeAccuracy(validY, valActivations)
            return pred, valAcc
        else:
            return pred, None

    def predict(self, X):
        # Input
        # X : Current Batch of Input Data as an nparray
        # Output
        # Returns the predictions made by the model (which are the activations output by the last layer)
        # Note: Activations at the first layer(input layer) is X itself
        activations = X
        for l in self.layers:
            activations = l.forwardpass(activations)
        return activations


class FullyConnectedLayer:
    def __init__(self, in_nodes, out_nodes, activation):
        # Method to initialize a Fully Connected Layer
        # Parameters
        # in_nodes - number of input nodes of this layer
        # out_nodes - number of output nodes of this layer
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.activation = activation
        # Stores a quantity that is computed in the forward pass but actually used in the backward pass. Try to identify
        # this quantity to avoid recomputing it in the backward pass and hence, speed up computation
        self.data = np.zeros((self.out_nodes, 1))

        # Create np arrays of appropriate sizes for weights and biases and initialise them as you see fit
        ###############################################
        # TASK 1a (Marks 0) - YOUR CODE HERE
        # raise NotImplementedError
        self.weights = np.random.rand(self.in_nodes, self.out_nodes)
        self.biases = np.zeros((1, self.out_nodes))
        ###############################################
        # NOTE: You must NOT change the above code but you can add extra variables if necessary

        # Store the gradients with respect to the weights and biases in these variables during the backward pass
        self.weightsGrad = np.zeros((self.out_nodes, self.in_nodes))
        self.biasesGrad = np.zeros((self.out_nodes, 1))

    def relu_of_X(self, X):
        # Input
        # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
        # Returns: Activations after one forward pass through this relu layer | shape: batchSize x self.out_nodes
        # This will only be called for layers with activation relu
        ###############################################
        # TASK 1b (Marks 1) - YOUR CODE HERE
        return np.maximum(np.zeros((X.shape[0], 1)), X)
        # raise NotImplementedError
        ####################################

    def gradient_relu_of_X(self, X, delta):
        # Input
        # data : Output from next layer/input | shape: batchSize x self.out_nodes
        # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
        # Returns: Current del_Error to pass to current layer in backward pass through relu layer | shape: batchSize x self.out_nodes
        # This will only be called for layers with activation relu amd during backwardpass
        ###############################################
        # TASK 1e (Marks 1) - YOUR CODE HERE
        X[X > 0] = 1
        X[X <= 0] = 0
        return delta*X
        # raise NotImplementedError
        ###############################################

    def softmax_of_X(self, X):
        # Input
        # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
        # Returns: Activations after one forward pass through this softmax layer | shape: batchSize x self.out_nodes
        # This will only be called for layers with activation softmax
        ###############################################
        # TASK 1c (Marks 3) - YOUR CODE HERE
        X=X-np.max(X,axis=1).reshape(X.shape[0],1)
        X[X<-700]=-700
        Xexp = np.exp(X)
        sum=(Xexp.sum(axis=1)).reshape(Xexp.shape[0],1)
        temp= np.divide(Xexp, sum)
        return temp
        # raise NotImplementedError
        ###############################################

    def gradient_softmax_of_X(self, X, delta):
        # Input
        # data : Output from next layer/input | shape: batchSize x self.out_nodes
        # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
        # Returns: Current del_Error to pass to current layer in backward pass through softmax layer | shape: batchSize x self.out_nodes
        # This will only be called for layers with activation softmax amd during backwardpass
        # Hint: You might need to compute Jacobian first
        ###############################################
        # TASK 1f (Marks 7) - YOUR CODE HERE
        grad=[]
        for data,delt in zip(X,delta):
            data=data.reshape(1,data.shape[0])
            delt=delt.reshape(1,delt.shape[0])
            grad.extend(delt@(np.diagflat(data)-data.T@data))
        grad=np.asarray(grad)
        return grad
        # raise NotImplementedError
        ###############################################

    def forwardpass(self, X):
        # Input
        # activations : Activations from previous layer/input | shape: batchSize x self.in_nodes
        # Returns: Activations after one forward pass through this layer | shape: batchSize x self.out_nodes
        # You may need to write different code for different activation layers
        ###############################################
        # TASK 1d (Marks 4) - YOUR CODE HERE
        if self.activation == 'relu':
            self.data = self.relu_of_X(np.dot(X, self.weights)+self.biases)
            return self.data
            # raise NotImplementedError
        elif self.activation == 'softmax':
            self.data=self.softmax_of_X(np.dot(X, self.weights)+self.biases)
            return self.data
            # raise NotImplementedError
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        ###############################################

    def backwardpass(self, activation_prev, delta):
        # Input
        # activation_prev : Output from next layer/input | shape: batchSize x self.out_nodes]
        # delta : del_Error/ del_activation_curr | shape: self.out_nodes
        # Output
        # new_delta : del_Error/ del_activation_prev | shape: self.in_nodes
        # You may need to write different code for different activation layers

        # Just compute and store the gradients here - do not make the actual updates
        ###############################################
        # TASK 1g (Marks 6) - YOUR CODE HERE
        if self.activation == 'relu':
            inp_delta=self.gradient_relu_of_X(self.data, delta)
            self.weightsGrad=activation_prev.T@inp_delta/activation_prev.shape[0]
            self.biasesGrad=np.sum(inp_delta,axis=0)/activation_prev.shape[0]
            # print(inp_delta,"relu")
            return inp_delta@self.weights.T
        elif self.activation == 'softmax':
            inp_delta=self.gradient_softmax_of_X(self.data, delta)
            self.weightsGrad=activation_prev.T@inp_delta/(delta.shape[0])
            self.biasesGrad=np.sum(inp_delta,axis=0)/delta.shape[0]
            # print(inp_delta,"softmax")
            return inp_delta@self.weights.T
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        ###############################################

    def updateWeights(self, lr):
        # Input
        # lr: Learning rate being used
        # Output: None
        # This function should actually update the weights using the gradients computed in the backwardpass
        ###############################################
        # TASK 1h (Marks 2) - YOUR CODE HERE
        self.weights-=lr*self.weightsGrad
        self.biases-=lr*self.biasesGrad
        # raise NotImplementedError
        ###############################################
