import numpy as np
import argparse
import csv
import matplotlib.pyplot as plt
''' 
You are only required to fill the following functions
mean_squared_loss
mean_squared_gradient
mean_absolute_loss
mean_absolute_gradient
mean_log_cosh_loss
mean_log_cosh_gradient
root_mean_squared_loss
root_mean_squared_gradient
preprocess_dataset
main

Don't modify any other functions or commandline arguments because autograder will be used
Don't modify function declaration (arguments)

'''

def mean_squared_loss(xdata, ydata, weights):
	'''
	weights = weight vector [D X 1]
	xdata = input feature matrix [N X D]
	ydata = output values [N X 1]
	Return the mean squared loss
	'''
	return np.sum((xdata@weights-ydata)**2)/(len(ydata))


def mean_squared_gradient(xdata, ydata, weights):
	'''
	weights = weight vector [D X 1]
	xdata = input feature matrix [N X D]
	ydata = output values [N X 1]
	Return the mean squared gradient
	'''
	return 2*xdata.T@(xdata@weights-ydata)

def mean_absolute_loss(xdata, ydata, weights):
	return np.sum(abs(xdata@weights-ydata))/len(ydata)
	raise NotImplementedError

def mean_absolute_gradient(xdata, ydata, weights):
	return xdata.T@np.sign(xdata@weights-ydata)
	raise NotImplementedError

def mean_log_cosh_loss(xdata, ydata, weights):
	return np.sum(np.log(np.cosh(np.minimum(np.full((len(ydata),1),100,dtype=float),abs(-ydata+xdata@weights))))/len(ydata))
	raise NotImplementedError

def mean_log_cosh_gradient(xdata, ydata, weights):
	return xdata.T@np.tanh(-ydata+xdata@weights)
	raise NotImplementedError

def root_mean_squared_loss(xdata, ydata, weights):
	return (np.sum((ydata-xdata@weights)**2)/len(ydata))**(0.5)
	raise NotImplementedError

def root_mean_squared_gradient(xdata, ydata, weights):
	return xdata.T@(xdata@weights-ydata)/((np.sum((ydata-xdata@weights)**2))**0.5)*len(ydata)**0.5
	raise NotImplementedError

class LinearRegressor:

	def __init__(self,dims):
		
		# dims is the number of the features
		# You can use __init__ to initialise your weight and biases
		# Create all class related variables here

		self.weights=np.ones((dims,1))

	def train(self, xtrain, ytrain, loss_function, gradient_function, epoch=100, lr=1.0):
		'''
		xtrain = input feature matrix [N X D]
		ytrain = output values [N X 1rgs.]
		learn weight vector [D X 1]
		epoch = scalar parameter epoch
		lr = scalar parameter learning rate
		loss_function = loss function name for linear regression training
		gradient_function = gradient name of loss function
		'''
		# temp=loss_function(xtrain,ytrain,self.weights)
		# You need to write the training loop to update weights here
		x=[]
		y=[]
		x.append(0)
		y.append(mean_squared_loss(xtrain,ytrain,self.weights))
		for i in range(1,epoch):
			self.weights=self.weights-lr*gradient_function(xtrain,ytrain,self.weights)/len(xtrain)
			x.append(i)
			y.append(mean_squared_loss(xtrain,ytrain,self.weights))
		# plt.plot(x,y)
			print(mean_squared_loss(xtrain,ytrain,self.weights))	
		

	def predict(self, xtest):
		
		# This returns your prediction on xtest
		return xtest@self.weights


def read_dataset(trainfile, testfile):
	'''
	Reads the input data from train and test files and 
	Returns the matrices Xtrain : [N X D] and Ytrain : [N X 1] and Xtest : [M X D] 
	where D is number of features and N is the number of train rows and M is the number of test rows
	'''
	xtrain = []
	ytrain = []
	xtest = []

	with open(trainfile,'r') as f:
		reader = csv.reader(f,delimiter=',')
		next(reader, None)
		for row in reader:
			xtrain.append(row[:-1])
			ytrain.append(row[-1])

	with open(testfile,'r') as f:
		reader = csv.reader(f,delimiter=',')
		next(reader, None)
		for row in reader:
			xtest.append(row)

	return np.array(xtrain), np.array(ytrain), np.array(xtest)

def preprocess_dataset(xdata, ydata=None):
	'''
	xdata = input feature matrix [N X D] 
	ydata = output values [N X 1]
	Convert data xdata, ydata obtained from read_dataset() to a usable format by loss function

	The ydata argument is optional so this function must work for the both the calls
	xtrain_processed, ytrain_processed = preprocess_dataset(xtrain,ytrain)
	xtest_processed = preprocess_dataset(xtest)	
	
	NOTE: You can ignore/drop few columns. You can feature scale the input data before processing further.
	'''
	temp=np.apply_along_axis(extract,1,xdata)
	xdata=np.append(xdata, temp , axis=1)
	xdata=np.delete(xdata, [0,1,2,5,7], axis=1)
	xdata=xdata.astype('float64')
	check=(xdata[:,6]!=0)*(xdata[:,6]!=3)
	change=np.average(xdata[:,6],weights=check)
	xdata[:,6][(xdata[:,6]==3) | (xdata[:,6]==0)]=change
	check=(xdata[:,5]!=0)*(xdata[:,5]!=3)
	change=np.average(xdata[:,5],weights=check)
	xdata[:,5][(xdata[:,5]==3) | (xdata[:,5]==0)]=change
	n_samples = len(xdata)
	mu = np.mean(xdata,axis=0)
	sigma = np.std(xdata, 0)
	for i in range(len(sigma)):
		if sigma[i]==0:
			sigma[i]=1
	xdata = (xdata-mu)/sigma
	xdata = np.hstack((np.ones((n_samples,1)),xdata))
	if ydata is not None:
		ydata=np.reshape(ydata,(len(ydata),1))
		ydata=ydata.astype('float64')
		return	xdata,ydata
	
	return xdata

dictionary_of_losses = {
	'mse':(mean_squared_loss, mean_squared_gradient),
	'mae':(mean_absolute_loss, mean_absolute_gradient),
	'rmse':(root_mean_squared_loss, root_mean_squared_gradient),
	'logcosh':(mean_log_cosh_loss, mean_log_cosh_gradient),
}

season={1:['1' ,'0' ,'0' ,'0' ],2:['0','1','0','0'],3:['0','0','1','0'],4:['0','0','0','1']}
week={'Monday':[1,0,0,0,0,0,0],'Tuesday':[0,1,0,0,0,0,0],'Wednesday':[0,0,1,0,0,0,0],'Thursday':[0,0,0,1,0,0,0],'Friday':[0,0,0,0,1,0,0],'Saturday':[0,0,0,0,0,1,0],'Sunday':[0,0,0,0,0,0,1]}
month={1:[1,0,0,0,0,0,0,0,0,0,0,0],2:[0,1,0,0,0,0,0,0,0,0,0,0],3:[0,0,1,0,0,0,0,0,0,0,0,0],4:[0,0,0,1,0,0,0,0,0,0,0,0],5:[0,0,0,0,1,0,0,0,0,0,0,0],6:[0,0,0,0,0,1,0,0,0,0,0,0],7:[0,0,0,0,0,0,1,0,0,0,0,0],8:[0,0,0,0,0,0,0,8,0,0,0,0],9:[0,0,0,0,0,0,0,0,1,0,0,0],10:[0,0,0,0,0,0,0,0,0,1,0,0],11:[0,0,0,0,0,0,0,0,0,0,1,0],12:[0,0,0,0,0,0,0,0,0,0,0,1]}
hour={}
day={}
for i in range(24):
	temp=[]
	for j in range(24):
		if j==i:
			temp.append(0)
		else:
			temp.append(1)
	hour[i]=temp
for i in range(1,32):
	temp=[]
	for j in range(1,32):
		if j==i:
			temp.append(0)
		else:
			temp.append(1)
	day[i]=temp
def extract(x):
	return [x[1][0:4]]  + season[int(x[7])] +season[int(x[2])] +week[x[5]]+month[int(x[1][5:7])]+ hour[int(x[3])] +day[int(x[1][8:10])]

def main():

	# You are free to modify the main function as per your requirements.
	# Uncomment the below lines and pass the appropriate value

	xtrain, ytrain, xtest = read_dataset(args.train_file, args.test_file)
	xtrainprocessed, ytrainprocessed = preprocess_dataset(xtrain, ytrain)
	xtestprocessed = preprocess_dataset(xtest)
	dims=np.size(xtrainprocessed,1)
	model = LinearRegressor(dims)
	# The loss function is provided by command line argument	
	loss_fn, loss_grad = dictionary_of_losses["rmse"]
	model.train(xtrainprocessed, ytrainprocessed, loss_fn, loss_grad, 10000, 0.1)

	# model1 = LinearRegressor(dims)
	# # The loss function is provided by command line argument	
	# loss_fn, loss_grad = dictionary_of_losses["mse"]
	# model1.train(xtrainprocessed, ytrainprocessed, loss_fn, loss_grad, 20000, 0.1)


	# model2 = LinearRegressor(dims)
	# # The loss function is provided by command line argument	
	# loss_fn, loss_grad = dictionary_of_losses["rmse"]
	# model2.train(xtrainprocessed, ytrainprocessed, loss_fn, loss_grad, 20000, 0.1)


	# model3 = LinearRegressor(dims)
	# # The loss function is provided by command line argument	
	# loss_fn, loss_grad = dictionary_of_losses["logcosh"]
	# model3.train(xtrainprocessed, ytrainprocessed, loss_fn, loss_grad, 20000, 0.1)
	
	# plt.gca().legend(('mae','mse','rmse','logcosh'))
	# plt.xlabel("No. of Epochs")
	# plt.ylabel("Mean Square Error")
	# plt.title("Different losses as a function of epoch")
	# # plt.gca().figure(num=None, figsize=(8, 6), dpi=8000, facecolor='w', edgecolor='k')
	# fig = plt.gcf()
	# fig.savefig('comparison.png', dpi=1000)
	# plt.show()
	ytest = model.predict(xtestprocessed)
	ytest=ytest.astype(int)
	with open('samplesubmission.csv', 'a') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerow(['instance (id)','count'])
		for i in range(len(ytest)):
			writer.writerow([i,max(ytest[i][0],0)])
	csvFile.close()


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--loss', default='mse', choices=['mse','mae','rmse','logcosh'], help='loss function')
	parser.add_argument('--lr', default=1.0, type=float, help='learning rate')
	parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
	parser.add_argument('--train_file', type=str, help='location of the training file')
	parser.add_argument('--test_file', type=str, help='location of the test file')

	args = parser.parse_args()
	main()