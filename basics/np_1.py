import numpy as np

#1. 배열 만들기
# np.array([]), np.arange(()), np.zeros(), np.ones()
# np.linspace(): 균등분할
print(np.array(range(1, 7, 2))) # [1 3 5]
print(np.arange(1, 7, 2))       # [1 3 5]
print(np.zeros([2, 3]))  #[[0. 0. 0.] [0. 0. 0]]
print(np.ones([2, 3]))   #[[1. 1. 1.] [1. 1. 1.]]
print(np.linspace(0, 1, 5))  #[0. 0.25 0.5 0.75 1.]

#2. 사칙연산
# 반복문 없이 연산
a1 = np.array([1, 2, 3])
b1 = np.array([12, 7, 9])

print(a1 + b1)
print(a1 - b1)
print(a1 * b1) # 1*12, 2*7, 3*9
print(a1 / b1) # 1/12, 2/7, 3/9

# list와 비교
li1 = [1, 2, 3]
arr1 = np.array([1, 2, 3])

print(li1 * 2)  # [1, 2, 3, 1, 2, 3]
print(arr1 * 2) # [2 4 6]
print(arr1 ** 2)  #[1 4 9]
# +, -, /도 됨
print(arr1 +2, arr1 - 2, arr1 / 2)

#3. A @ B == np.dot(A, B) vs. A * B
# 고차원에서는 @를 써야 함.
arr1 = np.array(range(1, 4))
arr2 = np.arange(4, 7)
print(arr1, arr2)  # [1 2 3] [4 5 6]

#4. 원소별 곱 "*"
print(arr1 * arr2)  # [4 10 18]

#5. 행렬 곱 "@"
print(arr1 @ arr2) # 1*4 + 2*5 + 3*6 = 32

# y = x @ w + b
y = arr1 @ arr2 + 1
print(y)

#6. 벡터 내적 / 외적
#6.1 내적: @, np.dot(a, b)
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

print(A @ B)
print(np.dot(A, B))

#6.2 외적1(3차원 벡터 외적): np.cross(a, b)
# 결과: x축과 y축에 수직인 z축 벡터가 나옴
print(np.cross(A, B))  #[(12 - 15) (12 - 6) (5 - 8)]

#6.3 외적2(모든 조합 곱): np.outer(a, b)
# 결과: 행렬 또는 펼쳐진 원소 조합
print(np.outer(A, B))
'''
[[ 4  5  6]         [[1*4 1*5 1*6
 [ 8 10 12]          [2*4 2*5 2*6
 [12 15 18]]         [3*4 3*5 3*6]]
'''

aa = np.array([[1, 2],
               [3, 4]])
bb = np.array([10, 20, 30])
print(np.outer(aa, bb))
'''
aa를 1차원으로 펼친 후 계산을 함.([1, 2, 3, 4] * [10, 20, 30])
[[ 10  20  30]      [[1*10 1*20 1*30]
 [ 20  40  60]       [2*10 2*20 2*60]
 [ 30  60  90]       [3*10 3*20 3*30]
 [ 40  80 120]]      [4*10 4*20 4*30]]
'''
print(np.multiply.outer(aa, bb))
'''
aa의 구조를 유지하고 aa의 각 원소에 bb 전체를 곱함
[[[ 10  20  30]         [[1 * bb,
  [ 20  40  60]]         [2 * bb]

 [[ 30  60  90]          [3 * bb,
  [ 40  80 120]]]        [4 * bb]]
'''

#7. 원하는 값 꺼내고 수정하기: 인덱싱 / 슬라이싱
#7.1 1차원
arr7 = np.array([10, 20, 30, 40, 50])

print(arr7[0])      # 10
print(arr7[1:4])    # [20 30 40]
print(arr1[-1])     # 3

#7.2 2차원
# arr[:, 열], arr[행, :], arr[시작:끝]
arr2 = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])
print(arr2)
print(arr2[0])      # [1 2 3]
print(arr2[0, 0])   # 1
print(arr2[1, 2])   # 6
print(arr2[:, 1])   # [2 5 8]
print(arr2[0, :])   # [1 2 3]

#8. 조건 처리와 Boolean indexing
#8.1
arr8 = np.array([10, 20, 30, 40, 50])

print(arr8 > 30)    # [False False False  True  True]
print(arr8[arr8 > 30]) # [40 50]
print(arr8)         # 원본 안 바뀜
print(arr8[[False, False, False, True, True]])
print(arr8[[arr8[0] > 30, arr8[1] > 30, arr8[2] > 30,
           arr8[3] > 30, arr8[4] > 30]])
# print(arr8[[20, 30, 40, 50, 70]]) #에러남. 불리언만 됨.

#8.2 Boolean index로 값 치환
arr8[arr8 >= 30] = 777
print(arr8)     # [ 10  20 777 777 777]

#8.3 Boolean index로 필터링
print(arr8[(arr8 > 10) & (arr8 < 777)])  # [20]
print(arr8[(arr8 == 10) | (arr8 == 20)]) # [10 20]

#9. 넘파이 파일로 저장하기
# 학습된 모델의 가중치 배열 저장용
np.save("np_test1", arr8) # (저장명, np 배열 객체)
np.savez_compressed("np_test2", array1=arr7, array2=arr8)

#10. 배열의 형태 바꾸기
# reshape(), flatten(), ravel(), transpose(), T
#19.1 CNN 예
#(1) 3x3 크기 흑백 이미지 2장 준비
images = np.array([
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ],
    [
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ]
])

print(images.shape) # (2, 3, 3) (이미지2장, 세로3픽셀, 가로3팩셀)

#(2) CNN용 이미지로 변환(이미지수, 높, 넓, 채널)
# 흑백이므로 1채녈: (2, 3, 3, 1)으로 reshape() -> 1줄로 만듦
images_reshaped = images.reshape(2, 3, 3, 1)
print(images_reshaped)
print(images_reshaped.shape)  # (2, 3, 3, 1)

#10.2 [기본 연습] 1행을 -> 여러 행으로 해보기
arr10_2 = np.arange(12)
print(arr10_2)
# [ 0  1  2  3  4  5  6  7  8  9 10 11]

arr10_2_1 = arr10_2.reshape(3, 4)
print(arr10_2_1)
'''
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
'''
arr_flat = arr10_2_1.flatten()
print(arr_flat)
print(arr10_2_1.T)
