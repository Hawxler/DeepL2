import tensorflow as tf
import numpy as np

t0 = tf.constant(1) # 스칼라 값(1)을 텐서로 만들기
print(t0)
print(tf.rank(t0)) # 몇 차원 텐서인지 확인

t1 = tf.constant([1, 2, 3]) # 리스트를 텐서로 만들기
print(t1)
print(tf.rank(t1))  # 몇 차원 텐서인지 확인

t2 = tf.constant([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
print(t2)
print(tf.rank(t2))

t3 = tf.constant([[[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]],
                 [[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]])
print(t3)
print(tf.rank(t3))

t3_1 = tf.stack([t2, t2])  # 기존 텐서 축적
print(t3_1)
print(tf.rank(t3_1))