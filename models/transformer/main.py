import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

# 데이터셋 클래스 정의
class TraceDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        # 'event_time'과 'Failed' 열 제외, 나머지를 입력 특징으로 사용
        self.features = self.data.drop(columns=['event_time', 'Failed'])
        self.labels = self.data['Failed'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.features.iloc[idx].values.astype(np.float32)
        label = self.labels[idx]
        return torch.tensor(features), torch.tensor(label, dtype=torch.float32)

# Transformer 모델 정의
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, output_size):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)  # 임베딩 레이어
        x = self.transformer_encoder(x)  # Transformer 인코더
        output = self.fc(x[:, -1, :])  # 마지막 타임스텝의 출력을 사용
        return output

# 파라미터 설정
input_size = 12  # 입력 특징 수 (입력 데이터의 열 수)
hidden_size = 64
num_heads = 4
num_layers = 2
output_size = 1  # 이진 분류 출력
learning_rate = 0.0005
num_epochs = 50

# 데이터셋 및 DataLoader 생성
dataset = TraceDataset('../../data/google_traces_v3/sampled_dataset.csv')
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 모델 초기화
model = TransformerModel(input_size, hidden_size, num_heads, num_layers, output_size)
criterion = nn.BCEWithLogitsLoss()  # 이진 분류를 위한 손실 함수
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 루프
model.train()
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs = inputs.unsqueeze(1)  # (batch_size, sequence_length=1, input_size)
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)

        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping 추가
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")