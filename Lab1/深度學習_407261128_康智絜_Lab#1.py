import matplotlib.pyplot as plt
import numpy as np

def train(w, dataset):
	res = None
	for x, label in dataset:
		x = np.array(x) # [1, x1, x2]
		if int( np.sign( w.T.dot(x)) ) != label:
			res = x, label
	return res

# learningRate -> Control w update adjustment range
def pla(dataset, _epoch, learningRate):
	w = np.zeros(3)
	epoch = 1
	while train(w, dataset) is not None:
		x, y = train(w, dataset)
		w += y * x * learningRate # w <- w + eta(yx)
		if epoch > _epoch:
			print ("epoch is > ", _epoch,", program has stopped to avoid from linearly inseparable")
			break;
		epoch += 1
	return w, epoch

def test(w, testset):
	_w = np.array([w[1], w[2]])
	for x in testset:
		_x = np.array([int(x[0]), int(x[1])])
		x.append( int(np.sign( _x.T.dot(_w) + w[0] )) )
	return testset

def draw(w, dataset, testResult):
	px1 = []	# dataset >0
	py1 = []
	px2 = []	# dataset <0
	py2 = []
	px3 = []	# testResult >0
	py3 = []
	px4 = []	# testResult <0
	py4 = []
	for x, c in dataset:
		if c == 1:
			px1.append(x[1])
			py1.append(x[2])
		else:
			px2.append(x[1])
			py2.append(x[2])
	for x in testResult:
		if x[2] == 1:
			px3.append(x[0])
			py3.append(x[1])
		else:
			px4.append(x[0])
			py4.append(x[1])
	size = max(max(px1), max(px2), max(px3), max(py1), max(py2), max(py3))
	x = np.linspace(-size, size, 10)
	y = (-w[0]-w[1]*x) / w[2]
	plt.figure()
	plt.plot(x, y)
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.plot(px1, py1, 'ko')
	plt.plot(px2, py2, 'rx')
	plt.plot(px3, py3, 'bs')
	plt.plot(px4, py4, 'gs')
	plt.show()

def main(_epoch, learningRate):
	# train
	file_train = open('train.txt', 'r')
	readData = file_train.readlines()
	dataset = []
	# store data to dataset
	for line in readData:
		x1, x2, label = line.replace('\n', ' ').split(',')
		dataset.append( [ (1, int(x1), int(x2)), (int(label)) ] )

	w, epoch = pla(dataset, _epoch, learningRate)

	if epoch < _epoch :
		print( 'w0 =' , w[0], ', w1 =', w[1],', w2 =' , w[2])

	# test
	file_test = open('test.txt', 'r')
	readTest = file_test.readlines()
	testset = []
	# store data to testset (for predicate)
	for line in readTest:
		x1, x2 = line.replace('\n',' ').split(',')
		testset.append( [int(x1), int(x2)] )
	testResult = test(w, testset)
	for i in testResult:
	 	print(i)

	draw(w, dataset, testResult)

if __name__ == '__main__':
	main(10000, 1)