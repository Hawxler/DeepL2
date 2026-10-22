import torch
from torch import nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(4, 4, bias=False)
        self.dropout = nn.Dropout(p=0.5)
        self.output = nn.Linear(4, 1, bias=False)

        # 초기 가중치 대입 과정의 기울기 추적만 끔
        with torch.no_grad():
            self.hidden.weight.copy_(torch.eye(4))
            self.output.weight.fill_(1.0)

    def forward(self, x):
        h = self.hidden(x)
        d = self.dropout(h)
        y = self.output(d)
        return h, d, y


# shape=(1, 4): 데이터 1개, 특성 4개
x = torch.tensor([[2.0, 4.0, 6.0, 8.0]])

model = SimpleModel()

# 학습 모드로 전환: Dropout 활성화
model.train()
h, d, y = model(x)

print("학습 모드")
print("Linear 출력:", h.detach().numpy())
print("Dropout 출력:", d.detach().numpy())
print("최종 출력:", y.detach().numpy())

# 평가·추론 모드로 전환: Dropout 비활성화
model.eval()
with torch.no_grad():
    h, d, y = model(x)

print("\n추론 모드")
print("Linear 출력:", h.numpy())
print("Dropout 출력:", d.numpy())
print("최종 출력:", y.numpy())