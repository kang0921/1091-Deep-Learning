import numpy as np
import warnings
warnings.filterwarnings("ignore", category=Warning)

learningRate = 0.01										# 學習率
tau = 0.1											# 容錯率
epoch = 30											# 最大世代數
num_InputX = 784										# 輸入層包含784個節點
num_hidden_layer = 1										# 隱藏層的層數
num_hidden_neuron = 30										# 隱藏層的神經元個數
num_OutputY = 3											# 輸出層包含三個神經元, 辨識 0, 1, 2 三個數字
num_train_img = 6000										# 要train的img的資料筆數
num_validate = 2000										# 用來驗證的img的資料筆數
num_test = 2000											# 無類別測試資料
output_model = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) 			# One-Hot Encoding
fault_tolerance = np.array([[tau], [tau], [tau]]) 						# 容錯率

# Initialize all network weights/biases to small random numbers 
# weight
layer1_weight = np.random.normal(loc=0.0, scale=0.25, size=(num_hidden_neuron, num_InputX) )	# 設 layer1 weight 有 num_hidden_neuron * 784 個在 -1 ~ 1 之間
layer2_weight = np.random.normal(loc=0.0, scale=0.25, size=(num_OutputY, num_hidden_neuron) )	# 設 layer2 weight 有 3 * num_hidden_neuron 個在 -1 ~ 1 之間
# bias
layer1_bias = np.random.normal(loc=0.0, scale=0.25, size=(num_hidden_neuron, 1) ) 		# 設 layer1 bias 有 num_hidden_neuron 個在 -1 ~ 1 之間
layer2_bias = np.random.normal(loc=0.0, scale=0.25, size=(num_OutputY, 1) ) 			# 設 layer2 bias 有 3 個在 -1 ~ 1 之間

weight = [0, layer1_weight, layer2_weight]
bias = [0, layer1_bias, layer2_bias]

# 得到訓練資料img的input X
def getInputX():
	file_img = open('train_img.txt', 'r')				 			# 讀檔
	readData = file_img.readlines()						 		# 讀取所有行
	InputX = []										# 宣告一個放input data的陣列

	for line in readData:
		readline = line.replace('\n', ' ').split(',')	 				# 分割字串,取得需要的數字(type為list)
		InputX.append( [int(r) for r in readline] )		 			# 將各個list中的input integer加入dataset中
	return np.array(InputX)

# 得到訓練資料img的Output Y
def getOutputY():
	file_label = open('train_label.txt', 'r')			 			# 讀檔
	readData = file_label.readlines()					 		# 讀取所有行
	OutputY = []										# 宣告一個放output label的陣列

	for line in readData:
		OutputY.append( int( line.replace('\n', ' ') ) ) 				# 分割字串,取得需要的數字(type為list)
	return np.array(OutputY)

# 得到測試資料img的input X
def getTestImg():
	file_test = open('test_img.txt', 'r')			 				# 讀檔
	readData = file_test.readlines()					 		# 讀取所有行
	InputX = []										# 宣告一個放input data的陣列

	for line in readData:
		readline = line.replace('\n', ' ').split(',')	 				# 分割字串,取得需要的數字(type為list)
		InputX.append( [int(r) for r in readline] )		 			# 將各個list中的input integer加入dataset中
	return np.array(InputX)

# activation function
def sigmoid(n):
	return 1.0 / ( 1.0 + np.exp( -n ) )

# loss function
def cross_entropy(y, a):
	# y 是 label
	# a 是 input neuron value
	return -( y * np.log( a ) + ( 1 - y ) * np.log( 1 - a ) )

#  計算訓練資料的錯誤度量的平均值
def average_cross_entropy(X, Y):
	sum_cross_entropy = np.array( [0.0]*num_OutputY ).reshape(num_OutputY, 1)		# [ [0.0], [0.0], [0.0] ]
	for i in range(num_train_img):
		a1, a2 = Feedforward(X, i)
		sum_cross_entropy += cross_entropy(output_model[ Y[i] ].reshape(num_OutputY, 1), a2)
	return sum_cross_entropy / num_train_img

# 前面6000筆的資料訓練的訓練準確率
def training_accuracy(X, Y):
	cnt = 0											# 計算訓練正確的筆數 
	for i in range(num_train_img):								# 第1~6000筆訓練資料
		a1, a2 = Feedforward(X, i)							# 取得a2
		row, column = np.where(a2 == np.max(a2)) 					# 取得訓練後預測的值(a2中的y1, y2, y3)最大的index
		y_hat = int(row)								# y^ = 預測的數字
		if Y[i] == y_hat:								# 判斷預測的數字是否等於正確的label
			cnt += 1								# 如果驗證相同則cnt加一
	return str(cnt *100 / np.double(num_train_img)) + " %"					# 印出訓練準確率

# 後面2000筆的資料訓練的驗證準確率
def vaildate_accuracy(X, Y):
	cnt = 0											# 計算訓練正確的筆數 
	for i in range(num_train_img, num_train_img + num_validate):				# 第6001~8000筆訓練資料
		a1, a2 = Feedforward(X, i)							# 取得a2
		row, column = np.where(a2 == np.max(a2)) 					# 取得訓練後預測的值(a2中的y1, y2, y3)最大的index
		y_hat = int(row)								# y^ = 預測的數字
		if Y[i] == y_hat:								# 判斷預測的數字是否等於正確的label
			cnt += 1								# 如果驗證相同則cnt加一
	return str(cnt *100 / np.double(num_validate)) + " %"					# 印出驗證準確率

# 測試資料預測結果
def test(X):
	testfile = open('test.txt', 'w')
	for i in range(num_test):
		a1, a2 = Feedforward(X, i)
		row, column = np.where(a2 == np.max(a2)) 					# 取得訓練後預測的值(a2中的y1, y2, y3)最大的index
		y_hat = int(row)								# y^ = 預測的數字
		testfile.write( str(y_hat) + '\n' )						# 寫入test.txt


# Compute the output for each neuron in the network
def Feedforward(X, i):
	# n^l = W^l*a^(l−1) + b^l and a^l = σ(n^l)
	a1 = sigmoid( weight[1].dot( X[i].reshape ( num_InputX, 1) ) + bias[1] ) 		# num_hidden_neuron * 1
	a2 = sigmoid( weight[2].reshape(num_OutputY, num_hidden_neuron).dot( a1 ) + bias[2] )	# 3 * 1

	return a1, a2

def Backward(a1, a2, X, Y, i, learningRate):
	global weight, bias

	# Step 2.1 Calculate the error vector for the output layer:
	delta_2 = a2 - output_model[ Y[i] ].reshape( num_OutputY, 1)				# 3 * 1

	# Step 2.2 Backpropagate the error for each hidden layer
	delta_1 = np.multiply( np.dot( weight[2].reshape( num_hidden_neuron, num_OutputY ), delta_2), np.multiply(a1, 1 - a1) )	# num_hidden_neuron * 1

	# Step 2.3 Update all of weight and bias values
	# layer 1
	weight[1] -= learningRate * delta_1 * X[i]						# num_hidden_neuron * 784
	bias[1] -= learningRate * delta_1							# num_hidden_neuron * 1
	# layer 2
	weight[2] -= learningRate * delta_2 * a1.reshape (1, num_hidden_neuron) 		# 3 * num_hidden_neuron
	bias[2] -= learningRate * delta_2							# 3 * 1

def stochastic_backpropagation(X, Y):

	now_epoch = 0
	while (average_cross_entropy(X, Y) >= fault_tolerance).all() and now_epoch < epoch:
		for i in range( num_train_img ):
			new_learningRate = learningRate * pow(0.95, now_epoch)			# 隨著世代增加，讓步伐逐漸減小
			a1, a2 = Feedforward(X, i)						# Feedforward
			Backward(a1, a2, X, Y, i, new_learningRate)				# Backward
		now_epoch += 1									# 增加一個世代

	print("停止於第 " + str(now_epoch) + " 世代數")					  # 印出停止於第幾個世代數



def main():
	X = getInputX()
	Y = getOutputY()
	stochastic_backpropagation(X, Y)
	test( getTestImg() )
	print("隱藏神經元的層數：" + str(num_hidden_layer))
	print("隱藏神經元的個數：" + str(num_hidden_neuron))
	print("學習率(learningRate)：" + str(learningRate))
	print("最大世代數(epoch)：" + str(epoch))
	print("訓練準確率：" + training_accuracy(X, Y))
	print("驗證準確率：" + vaildate_accuracy(X, Y))

if __name__ == '__main__':
	main()
