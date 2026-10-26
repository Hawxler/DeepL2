"""두 PyTorch 모델이 공유하는 CPU 학습·평가·추론 함수."""
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def train(model, data, epochs=200, batch_size=16, lr=0.001):
    dataset = TensorDataset(torch.from_numpy(data.X_train),
                            torch.from_numpy(data.y_train))
    # BatchNorm1d의 학습 배치에는 샘플이 2개 이상 필요합니다.
    if batch_size < 2 or len(dataset) % batch_size == 1:
        raise ValueError('BatchNorm 때문에 마지막 배치까지 2개 이상이어야 합니다. batch_size=16을 사용하세요.')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0.0, 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()  # 이전 배치의 기울기를 초기화
            logits = model(batch_x)
            # float32 원-핫 정답도 CrossEntropyLoss에서 지원합니다.
            loss = loss_fn(logits, batch_y)
            loss.backward()  # 역전파로 기울기 계산
            optimizer.step()  # 기울기에 따라 가중치 갱신
            total_loss += loss.item() * len(batch_x)
            correct += (logits.argmax(1) == batch_y.argmax(1)).sum().item()
        result = {'loss': total_loss / len(dataset), 'accuracy': correct / len(dataset)}
        history.append(result)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"Epoch {epoch + 1}/{epochs} - loss: {result['loss']:.4f} - accuracy: {result['accuracy']:.4f}")
    return history


def evaluate(model, data):
    model.eval()  # Dropout을 끄고 BatchNorm의 학습된 통계 사용
    with torch.no_grad():
        logits = model(torch.from_numpy(data.X_test))
        targets = torch.from_numpy(data.y_test)
        loss = nn.CrossEntropyLoss()(logits, targets).item()
        accuracy = (logits.argmax(1) == targets.argmax(1)).float().mean().item()
    return {'loss': loss, 'accuracy': accuracy}


def predict(model, data, rows):
    model.eval()
    with torch.no_grad():  # 추론에서는 기울기 기록 불필요
        logits = model(torch.from_numpy(data.transform(rows)))
        probabilities = torch.softmax(logits, dim=1).numpy()
    names = data.class_names[probabilities.argmax(axis=1)]
    return names, probabilities
