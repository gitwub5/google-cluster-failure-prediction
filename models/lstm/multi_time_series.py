import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# 데이터셋 클래스 정의
class TraceDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data.sort_values(by=['event_time'], inplace=True)  # 정렬
        self.features = self.data.drop(columns=['collection_id', 'event_time', 'Failed', 'machine_id', 'event_type'])
        self.labels = self.data['Failed'].values
        self.event_types = self.data['event_type'].values

        # 고유 event_type을 정수 인덱스로 변환
        self.event_type_to_idx = {etype: idx for idx, etype in enumerate(self.data['event_type'].unique())}
        self.data['event_type_idx'] = self.data['event_type'].map(self.event_type_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.features.iloc[idx].values.astype(np.float32)
        label = self.labels[idx]
        event_type_idx = self.data.iloc[idx]['event_type_idx']
        return (
            torch.tensor(features),
            torch.tensor(event_type_idx, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )

class SlidingWindowDataset(Dataset):
    def __init__(self, csv_file, window_size=10):
        self.data = pd.read_csv(csv_file)
        self.data.sort_values(by=['event_time'], inplace=True)  # 정렬

        # 고유 event_type을 정수 인덱스로 변환
        self.event_type_to_idx = {etype: idx for idx, etype in enumerate(self.data['event_type'].unique())}
        self.data['event_type_idx'] = self.data['event_type'].map(self.event_type_to_idx)

        # 슬라이딩 윈도우 생성
        self.window_size = window_size
        self.sequences, self.targets = self._create_sliding_windows()

    def _create_sliding_windows(self):
        sequences = []
        targets = []

        for _, group in self.data.groupby('event_type_idx'):
            machine_features = group.drop(
                columns=['Failed', 'machine_id', 'event_type', 'collection_id', 'event_time', 'event_type_idx']
            ).values
            machine_labels = group['Failed'].values
            event_types = group['event_type_idx'].values

            for i in range(len(machine_features) - self.window_size):
                sequences.append((machine_features[i:i + self.window_size], event_types[i]))
                targets.append(machine_labels[i + self.window_size])

        return sequences, targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, event_type_idx = self.sequences[idx]
        target = self.targets[idx]
        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(event_type_idx, dtype=torch.long),
            torch.tensor(target, dtype=torch.float32),
        )

# 모델 정의
class MultiTimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_event_types, embedding_dim_event):
        super(MultiTimeSeriesModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Event Type Embedding Layer
        self.event_embedding = nn.Embedding(num_event_types, embedding_dim_event)

        # LSTM for Time Series Data
        self.lstm = nn.LSTM(input_size + embedding_dim_event, hidden_size, num_layers, batch_first=True)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, event_type_idx):
        event_embeddings = self.event_embedding(event_type_idx)

        # Embedding 차원 맞추기
        event_embeddings = event_embeddings.unsqueeze(1).repeat(1, x.size(1), 1)

        # 입력 데이터 결합
        x = torch.cat((x, event_embeddings), dim=2)

        # LSTM Forward
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        output, _ = self.lstm(x, (h_0, c_0))
        output = self.fc(output[:, -1, :])  # 마지막 타임스텝 사용
        return output

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 및 DataLoader 생성
    window_size = 10
    dataset = SlidingWindowDataset('../../data/google_traces_v3/filtered_dataset.csv', window_size=window_size)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 파라미터 설정
    input_size = 10
    hidden_size = 64
    num_layers = 1
    output_size = 1
    num_event_types = dataset.data['event_type'].nunique()
    embedding_dim_event = 2
    learning_rate = 0.001
    num_epochs = 200

    # 모델 초기화
    model = MultiTimeSeriesModel(
        input_size, hidden_size, num_layers, output_size,
        num_event_types, embedding_dim_event
    ).to(device)

    failed_count = sum([1 for target in dataset.targets if target == 1])
    success_count = sum([1 for target in dataset.targets if target == 0])
    pos_weight = torch.tensor([success_count / failed_count]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        all_preds = []
        all_labels = []
        running_loss = 0.0

        for sequences, event_types, labels in train_loader:
            sequences = sequences.to(device)
            event_types = event_types.to(device)
            labels = labels.to(device)

            outputs = model(sequences, event_types)
            loss = criterion(outputs.squeeze(), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs.squeeze()) > 0.5).int()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    print("Training complete.")

    # 모델 저장
    torch.save(model.state_dict(), './models/multi_time_series_model.pth')
    print(f"Model saved.")

    # 결과 시각화
    plt.figure(figsize=(16, 8))
    #
    # # 손실
    # plt.subplot(2, 2, 1)
    # plt.plot(range(1, num_epochs + 1), epoch_losses, label='Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')
    # plt.legend()
    #
    # # 정확도
    # plt.subplot(2, 2, 2)
    # plt.plot(range(1, num_epochs + 1), epoch_accuracies, label='Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Training Accuracy')
    # plt.legend()
    #
    # # 정밀도와 재현율
    # plt.subplot(2, 2, 3)
    # plt.plot(range(1, num_epochs + 1), epoch_precisions, label='Precision')
    # plt.plot(range(1, num_epochs + 1), epoch_recalls, label='Recall')
    # plt.xlabel('Epoch')
    # plt.ylabel('Score')
    # plt.title('Precision and Recall')
    # plt.legend()
    #
    # # F1 점수
    # plt.subplot(2, 2, 4)
    # plt.plot(range(1, num_epochs + 1), epoch_f1_scores, label='F1-Score')
    # plt.xlabel('Epoch')
    # plt.ylabel('F1-Score')
    # plt.title('Training F1-Score')
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()
