"""PyTorch의 Sequential / forward() 직접 작성 모델."""
from torch import nn


def build_sequential():
    return nn.Sequential(
        nn.Linear(4, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(16, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 3),
    )


class IrisModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(4, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.hidden2 = nn.Linear(16, 8)
        self.bn2 = nn.BatchNorm1d(8)
        self.output = nn.Linear(8, 3)

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.hidden1(x))))
        x = self.relu(self.bn2(self.hidden2(x)))
        # CrossEntropyLoss가 내부에서 log_softmax를 처리하므로 logits 반환.
        return self.output(x)
