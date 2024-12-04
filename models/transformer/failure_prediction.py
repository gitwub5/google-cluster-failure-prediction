import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 데이터셋 클래스 정의
class SequenceDataset(Dataset):
    def __init__(self, data, sequence_length, features, target_col='Failed'):
        self.data = data
        self.sequence_length = sequence_length
        self.features = features
        self.target_col = target_col
        self.machine_sequences = {}  # machine_id별 시퀀스 저장
        self.machine_ids = []
        self.prepare_data()

    def prepare_data(self):
        grouped = self.data.groupby('machine_id')
        for machine_id, group in grouped:
            values = group[self.features].values  # 여러 특성의 값 추출
            targets = group[self.target_col].values
            event_types = group['event_type_idx'].values

            sequences, event_labels, sequence_targets = [], [], []
            for i in range(len(values) - self.sequence_length):
                sequences.append(values[i:i + self.sequence_length])
                event_labels.append(event_types[i:i + self.sequence_length])
                sequence_targets.append(targets[i + self.sequence_length])

            self.machine_sequences[machine_id] = {
                'sequences': sequences,
                'event_labels': event_labels,
                'targets': sequence_targets
            }

            self.machine_ids.extend([machine_id] * len(sequences))

    def __len__(self):
        return sum(len(seq_data['sequences']) for seq_data in self.machine_sequences.values())

    def __getitem__(self, idx):
        machine_ids = list(self.machine_sequences.keys())
        for machine_id in machine_ids:
            seq_data = self.machine_sequences[machine_id]
            if idx < len(seq_data['sequences']):
                return (
                    torch.tensor(seq_data['sequences'][idx], dtype=torch.float32),
                    torch.tensor(seq_data['event_labels'][idx], dtype=torch.long),
                    torch.tensor(seq_data['targets'][idx], dtype=torch.float32),
                    machine_id
                )
            idx -= len(seq_data['sequences'])

# Transformer 모델 정의
class TransformerModel(nn.Module):
    def __init__(self, input_size, embedding_dim, num_heads, num_layers, output_size, num_event_types):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(num_event_types, embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size + embedding_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size + embedding_dim, output_size)

    def forward(self, x, event_type_idx):
        event_embedding = self.embedding(event_type_idx)  # (batch_size, seq_len, embedding_dim)
        x = torch.cat((x, event_embedding), dim=2)  # (batch_size, seq_len, input_size + embedding_dim)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, input_size + embedding_dim)
        x = x[:, -1, :]  # 마지막 시점 사용
        output = self.fc(x)  # (batch_size, output_size)
        return output

# 학습 및 평가 함수
def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for sequences, event_labels, targets, _ in dataloader:
            sequences, event_labels, targets = sequences.to(device), event_labels.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(sequences, event_labels)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

def evaluate_model(model, dataloader, device):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for sequences, event_labels, targets in dataloader:
            sequences, event_labels, targets = sequences.to(device), event_labels.to(device), targets.to(device)
            outputs = model(sequences, event_labels)
            preds = torch.round(torch.sigmoid(outputs.squeeze()))  # 이진 분류
            predictions.extend(preds.cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions)
    recall = recall_score(actuals, predictions)
    f1 = f1_score(actuals, predictions)
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# 모델 저장 함수
def save_model(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# 모델 로드 함수
def load_model(model, load_path, device):
    model.load_state_dict(torch.load(load_path, map_location=device))
    print(f"Model loaded from {load_path}")
    return model

# 실행
if __name__ == "__main__":
    # 데이터 준비
    file_path = '../../data/google_traces_v3/train_data.csv'
    data = pd.read_csv(file_path)

    # 전처리
    data['event_type_idx'] = pd.Categorical(data['event_type']).codes
    print("Train event_type_idx:", data['event_type_idx'].unique())

    sequence_length = 24
    features = ['average_usage_cpus', 'average_usage_memory', 'maximum_usage_cpus', 'maximum_usage_memory']
    dataset = SequenceDataset(data, sequence_length, features, target_col='Failed')

    # DataLoader 준비
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Transformer 모델 설정
    input_size = len(features)
    embedding_dim = 16
    num_heads = 4
    num_layers = 2
    output_size = 1  # 이진 분류
    num_event_types = data['event_type_idx'].nunique()

    model = TransformerModel(input_size, embedding_dim, num_heads, num_layers, output_size, num_event_types)

    # 가중치 계산 및 손실 함수 설정
    num_failed = (data['Failed'] == 1).sum()  # 실패한 경우의 수
    num_not_failed = (data['Failed'] == 0).sum()  # 실패하지 않은 경우의 수
    class_weight = torch.tensor([num_not_failed / num_failed], dtype=torch.float32)  # 실패에 대한 가중치
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)  # 가중치 적용

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 학습
    train_model(model, dataloader, criterion, optimizer, num_epochs=20, device=device)

    # 모델 저장
    model_save_path = './models/transformer_failure_model.pth'
    save_model(model, model_save_path)

    # 모델 로드
    model = TransformerModel(input_size, embedding_dim, num_heads, num_layers, output_size, num_event_types).to(device)
    model = load_model(model, model_save_path, device)

    # 평가
    evaluate_model(model, dataloader, device)