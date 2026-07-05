import matplotlib, matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r"0_data\test4.csv")
print(df.head(3))

x = df['number']
y = df['height']

# plt.plot(x)
# plt.show()

# plt.plot(y)
# plt.show()

# plt.plot(x, y) # x-y 그래프
# plt.show()

# plt.plot(x, y)
# plt.axis([0, 20, 100, 200]) #x축: 0~10, y축: 100~200
# plt.show()

# plt.plot(x, y, "#FFE400")  # 'k': black, 'g': green
# plt.show()

# # 그래프 추가
# plt.plot(x, y, 'r', x, y+10, 'g')
# plt.show()

# # 산점도: 
# # 'o'는 점 표시, 's' 네모
# # 'rs': red square, 'go': green o, 'r*
# plt.plot(x, y, 'r', x, y+10, 'g*')
# plt.show()

# plt.plot(x, y, 'rs', x, y+10, 'go')
# plt.show()

# # 범례(Legend)
# plt.plot(x, y, 'r', label='height1')
# plt.plot(x, y+10, 's', label='height2')
# plt.legend()
# plt.xlabel('number')
# plt.show()

# # 범례 위치: 
# # loc=best 최적화, 
# plt.plot(x, y, 'g', label='height1')
# plt.plot(x, y+10, 'rs', label='height2')
# plt.legend(loc='best', fontsize=20)
# plt.xlabel('number')
# plt.ylabel('height')
# plt.show()

# # c(색)은 Hex code 사용
# plt.scatter(x, y, c='#E389E9')
# plt.show()

# 
plt.scatter(x, y, c='#E389E9', label='height1')
plt.scatter(x, y+10, label='height2')
plt.legend(loc='best', fontsize=15)
plt.xlabel('number')
plt.ylabel('height')
plt.show()