import matplotlib.pyplot as plt
import numpy as np
import random
import math

# x0, x1, x2, y
data = np.array([
((1, 0, 0), 0),
((1, 0, 1), 1),
((1, 1, 0), 1),
((1, 1, 1), 1)], dtype = object)

def sigmoid(n):
	return 1.0 / (1.0 + math.exp(-n))

def init_weight():
	w = np.array( [np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
	print("Initial weight is ", w)
	return w

def Logistic_Regression(data, epoch, eta, init):
	w = init.copy()
	for i in range(epoch):
		for x, y in data:
			x = np.array(x) # [1, x1, x2]
			w += eta * (y - sigmoid( w.T.dot(x) )) * x		# w <- w + eta(y-^y)x
	return w

def draw(w, data, init):
	px1 = []	# dataset >0
	py1 = []
	px2 = []	# dataset <0
	py2 = []
	for x, y in data :
		if y == 1:
			px1.append(x[1])
			py1.append(x[2])
		else:
			px2.append(x[1])
			py2.append(x[2])

	size = max(max(px1), max(px2), max(py1), max(py2))
	x = np.linspace(-size, size, 10)
	y1 = (-w[0]-w[1]*x) / w[2]
	y2 = (-init[0]-init[1]*x) / init[2]

	plt.figure()
	plt.plot(x, y1, label = 'Train')	#畫Train線
	plt.plot(x, y2, label = 'Init', color = 'red', linestyle = '--')		#畫Init線

	plt.title('Training and Testing Data', size = 20)	# 標題

	plt.xlabel('x1')	# x軸座標
	plt.ylabel('x2')	# y軸座標

	l1 = plt.plot(px1, py1, 'ko', label = '1 (Training)')	# 畫點(y = 1)
	l2 = plt.plot(px2, py2, 'rx', label = '0 (Training)')	# 畫點(y = 0)

	plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')	#圖例
	plt.tight_layout()
	plt.show()

def main(epoch, eta):
	init = init_weight()
	w = Logistic_Regression(data, epoch, eta, init)
	print( '\nw0 =' , w[0], '\nw1 =', w[1],'\nw2 =' , w[2])
	draw(w, data, init)

if __name__ == '__main__':
	main(10000,0.2)