import matplotlib.pyplot as plt
import numpy as np
import random
import warnings

#為了ignore warning
warnings.filterwarnings("ignore", category = Warning )

# x0, x1, x2, y
data = np.array([
((1, 0, 0), 0),
((1, 0, 1), 0),
((1, 1, 0), 0),
((1, 1, 1), 1)], dtype = object)

def init_weight():
	w = np.random.uniform(-1.0, 1.0, 3)
	print("\nInitial weight is [ %f, %f, %f]" % (w[0], w[1], w[2]))
	return w

def sigmoid(n):
	return 1.0 / (1.0 + np.exp(-n))

def Cross_Entropy(y, y_Hat):
	return -( y * np.log( y_Hat ) + (1 - y ) * np.log( 1 - y_Hat ) ) 

def Gradient_Descent(data, epoch, LearningRate, init, Error):
	w = init.copy() 
	count = 0
	_error = 1
	while count <= epoch:
		if _error < Error:
			break
		_error = 0
		for x, y in data:
			x = np.array(x) # x = [1, x1, x2]
			w += LearningRate * (y - sigmoid( w.T.dot(x) )) * x	# w <- w + LearningRate(y-^y)x
			_error += Cross_Entropy(y, sigmoid( w.T.dot(x)))
		count += 1
		_error /= len(data)

	print("\nThe maximum number of epoches is", count)
	return w

def draw(w, data, init):
	train_x1 = []	# dataset > 0
	train_y1 = []
	train_x2 = []	# dataset < 0
	train_y2 = []
	for x, y in data :
		if y == 1:
			train_x1.append(x[1])
			train_y1.append(x[2])
		else:
			train_x2.append(x[1])
			train_y2.append(x[2])


	size = max( max(train_x1), max(train_y1), 
				max(train_x2), max(train_y2))
	X = np.linspace(-size, size, 10)

	train_Y = (-w[0]-w[1]*X) / w[2]
	init_Y = (-init[0]-init[1]*X) / init[2]

	plt.figure()
	plt.plot(X, train_Y, label = 'Train', color = 'c')	#畫Train線
	plt.plot(X, init_Y, label = 'Init', color = 'red', linestyle = '--')		#畫Init線

	plt.title('Case 1', size = 20)	# 標題

	plt.xlabel('x1', size = 12, labelpad = 10)	# x軸
	plt.ylabel('x2', size = 12, labelpad = 10, rotation = 'horizontal')	# y軸

	plt.plot(train_x1, train_y1, 'ko', label = '1 (Training)')	# 畫點(y = 1)
	plt.plot(train_x2, train_y2, 'rx', label = '0 (Training)')	# 畫點(y = 0)

	plt.legend( loc='best')	#圖例
	plt.show()

def main(epoch, LearningRate, Error):
	print("Learning Rate is", LearningRate)
	print("\nError limit is", Error)
	init = init_weight()
	w = Gradient_Descent(data, epoch, LearningRate, init, Error)
	print("\nNew weight is [ %f, %f, %f]" % (w[0], w[1], w[2]))
	draw(w, data, init)
	
if __name__ == '__main__':
	main(100000, 0.3, 0.01)