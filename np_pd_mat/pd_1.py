import pandas as pd

import pandas as pd

s = pd.Series([10, 20, 30, 40])
print(s)

###########################
'''
df.head()        # 앞 5개 행 보기
df.tail()        # 뒤 5개 행 보기
df.info()        # 컬럼, 결측치, 자료형 정보
df.describe()    # 숫자 데이터 통계 요약
df.shape         # 행, 열 개수
df.columns       # 컬럼 이름 확인
df.index         # 인덱스 확인
df.dtypes        # 각 컬럼의 자료형 확인
'''
###########################

df = pd.read_csv(r"0_data\test4.csv")
print(type(df))
#1. 전반 정보
print(df.info())
#2. 데이터 형태 보기: 첫 5개
print(df.head())
print(df.tail())
#3. 무슨 정보가 있는지 보기: 칼럼 헤드
print(df.columns)
#4. 각 칼럼의 자료형 확인
print(df.dtypes)
#5. 숫자 정보 요약: 최대, 최소, 평균, 분산(standard diviation)
print(df.describe())
#6. 몇 행 몇 열?
print(df.shape)
#7. 인덱스 확인
print(df.index)

