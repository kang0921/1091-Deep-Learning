import matplotlib.pyplot as plt
import numpy as np
import random
import warnings

#為了ignore warning
warnings.filterwarnings("ignore", category = Warning )

# x0, x1, x2, y
data = np.array([
((1, 170, 80), 1),
((1,  90, 15), 0),
((1, 130, 30), 0),
((1, 165, 55), 1),
((1, 150, 45), 1),
((1, 120, 40), 0),
((1, 110, 35), 0),
((1, 180, 70), 1),
((1, 175, 65), 1),
((1, 160, 60), 1)] , dtype = object)

predict_Data = np.array([
(1, 170, 60),
(1,  85, 15),
(1, 145, 45)], dtype = object)

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

def predict(predict_Data, w):
	predict_result = []
	print("\nPredict testing example are")
	cnt = 0
	for i in predict_Data:
		cnt += 1
		label = 1 if sigmoid(w.T.dot(i)) >= 0.5 else 0
		predict_result.append([(i[0], i[1], i[2]), label])
		print('Case %d ： ( %d , %d , %d )' % ( cnt, i[1], i[2], label))
	return predict_result

def draw(w, data, init, predict_result):
	train_x1 = []	# dataset > 0
	train_y1 = []
	train_x2 = []	# dataset < 0
	train_y2 = []
	test_x1 = []	# predict_result > 0
	test_y1 = []
	test_x2 = []	# predict_result < 0
	test_y2 = []
	for x, y in data :
		if y == 1:
			train_x1.append(x[1])
			train_y1.append(x[2])
		else:
			train_x2.append(x[1])
			train_y2.append(x[2])
	for x, y in predict_result :
		if y == 1:
			test_x1.append(x[1])
			test_y1.append(x[2])
		else:
			test_x2.append(x[1])
			test_y2.append(x[2])

	X = np.linspace(-10, 200, 10)

	train_Y = (-w[0]-w[1]*X) / w[2]
	init_Y = (-init[0]-init[1]*X) / init[2]

	plt.figure()
	plt.plot(X, train_Y, label = 'Train', color = 'c')	#畫Train線
	plt.plot(X, init_Y, label = 'Init', color = 'red', linestyle = '--')		#畫Init線

	plt.title('Training and Testing Data', size = 20)	# 標題

	plt.xlabel('x1', size = 12, labelpad = 10)	# x軸
	plt.ylabel('x2', size = 12, labelpad = 10, rotation = 'horizontal')	# y軸

	plt.plot(train_x1, train_y1, 'ko', label = '1 (Training)')	# 畫點(y = 1)
	plt.plot(train_x2, train_y2, 'rx', label = '0 (Training)')	# 畫點(y = 0)
	plt.plot(test_x1, test_y1, 'b^', label = '1 (Testing)')	# 畫點(y = 1)
	plt.plot(test_x2, test_y2, 'g^', label = '0 (Testing)')	# 畫點(y = 0)

	plt.legend( loc='best')	#圖例
	plt.show()

def main(epoch, LearningRate, Error):
	print("Learning Rate is", LearningRate)
	print("\nError limit is", Error)
	init = init_weight()
	w = Gradient_Descent(data, epoch, LearningRate, init, Error)
	print("\nNew weight is [ %f, %f, %f]" % (w[0], w[1], w[2]))
	predict_result = predict(predict_Data, w)
	draw(w, data, init, predict_result)
	
if __name__ == '__main__':
	main(100000, 0.5, 0.01)