import tensorflow as tf

# # Tensor 자료형
# tensor1 = tf.constant(3)
# print(tensor1)   # tf.Tensor(3, shape=(), dtype=int32)

# # 사칙연산 
# tensor1 = tf.constant([3, 4, 5])
# tensor2 = tf.constant([1, 2, 3])
# print(tensor1 + tensor2)  # tf.Tensor([4 6 8], shape=(3,), dtype=int32)
# print(tensor1 * tensor2)  # tf.Tensor([3 8 15], shape=(3,), dtype=int32)
# print(tensor1 / tensor2)  # tf.Tensor([3. 2. 1.66666667], shape=(3,), dtype=float64)

# # 사칙연산함수
# # tf.add(), tf.subtract(), tf.divide(), tf.multiply()
# # tf.matmul() 행렬의 곱
# tensor1 = tf.constant([3, 4, 5])
# tensor2 = tf.constant([1, 2, 3])
# print(tf.add(tensor1, tensor2))
# print(tf.multiply(tensor1, tensor2))

# tensor3 = tf.constant([[1, 2],
#                       [3, 4]])
# tensor4 = tf.constant([[1, 5],
#                       [3, 7]])
# print(tf.matmul(tensor3, tensor4))
'''
tf.Tensor(
[[ 7 19]       # 1*1 + 2*3 = 7, 1*5 + 2*7=19
 [15 43]], shape=(2, 2), dtype=int32)
'''

# # 텐서 모양(shape) 보기 
# tensor1 = tf.constant([3, 4, 5])
# print(tensor1.shape)             # (3,)
# tensor3 = tf.constant([[3, 4],
#                        [2, 1]])  # (2, 2)
# print(tensor3.shape)
# tensor5 = tf.zeros([2, 2, 3])    # (2, 2, 3)
# print(tensor5.shape)

# 텐서 자료형
# 주: weight는 Variable이어야 함.
w = tf.Variable(1.0)   # tf.Variable: TF의 변수 객체
print(w)  # <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>
'''
w = 0.98 -> 0.95 -> 0.91... 처럼 w는 계속 변해야 함.
따라서, tf.constant()는 상수라서 적합하지 않고 Variable이어야 함.
'''

# 변수값 변경
w = tf.Variable(1.1)    # 변수값을 1.1로 초기화함.
w.assign(2.1)           # 변수값을 2.1로 바는 함수.
print(w.numpy())   # 2.1
print(w)  # <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.1>