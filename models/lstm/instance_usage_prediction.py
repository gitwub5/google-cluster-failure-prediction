import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# 성능 지표 저장용 딕셔너리 선언
epoch_metrics = {
    "epoch": [],
    "loss": [],
    "mse": [],
    "rmse": [],
    "r2": []
}


# 데이터셋 클래스 정의
class SequenceDataset(Dataset):
    def __init__(self, data, sequence_length, features):
        self.data = data
        self.sequence_length = sequence_length
        self.features = features
        self.machine_sequences = {}  # machine_id별 시퀀스 저장
        self.prepare_data()

    def prepare_data(self):
        grouped = self.data.groupby('machine_id')  # machine_id별 데이터 그룹화

        for machine_id, group in grouped:
            values = group[self.features].values  # 여러 특성의 값 추출
            event_types = group['event_type_idx'].values

            sequences, targets, event_labels = [], [], []
            for i in range(len(values) - self.sequence_length):
                sequences.append(values[i:i + self.sequence_length])
                targets.append(values[i + self.sequence_length])
                event_labels.append(event_types[i + self.sequence_length])

            self.machine_sequences[machine_id] = {
                'sequences': sequences,
                'targets': targets,
                'event_labels': event_labels
            }

    def get_machine_dataset(self, machine_id):
        if machine_id not in self.machine_sequences:
            raise ValueError(f"Machine ID {machine_id} not found in dataset.")
        data = self.machine_sequences[machine_id]

        # 각 데이터 변환
        sequences = torch.tensor(np.array(data['sequences']), dtype=torch.float32)
        targets = torch.tensor(np.array(data['targets']), dtype=torch.float32)
        event_labels = torch.tensor(np.array(data['event_labels']), dtype=torch.long)

        # 데이터 길이 확인
        assert len(sequences) == len(targets) == len(event_labels), \
            "Lengths of sequences, targets, and event_labels must match."

        return TensorDataset(sequences, targets, event_labels)


# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_event_types, event_embedding_dim):
        """
        LSTM 기반 성능 예측 모델
        """
        super(LSTMModel, self).__init__()

        # Event Type Embedding Layer
        self.event_embedding = nn.Embedding(num_event_types, event_embedding_dim)

        # LSTM Layer
        self.lstm = nn.LSTM(input_size + event_embedding_dim, hidden_size, num_layers, batch_first=True)

        # Fully Connected Layer for Regression
        self.fc_regression = nn.Linear(hidden_size, output_size)

    def forward(self, x, event_type_idx):
        """
        모델의 Forward Pass
        """
        # Event Type Embedding
        event_embedding = self.event_embedding(event_type_idx)  # (batch_size, embedding_dim)
        event_embedding = event_embedding.unsqueeze(1).repeat(1, x.size(1), 1)

        # 입력 데이터와 이벤트 임베딩 결합
        x = torch.cat((x, event_embedding), dim=2)

        # LSTM Forward
        _, (hidden, _) = self.lstm(x)

        # Regression Output
        regression_output = self.fc_regression(hidden[-1])  # (batch_size, output_size)

        return regression_output


def prepare_dataloaders(data, sequence_length, batch_size, features):
    dataset = SequenceDataset(data, sequence_length, features)
    dataloaders = {}
    for machine_id in dataset.machine_sequences.keys():
        if len(dataset.machine_sequences[machine_id]['sequences']) == 0:
            continue
        machine_dataset = dataset.get_machine_dataset(machine_id)
        dataloaders[machine_id] = DataLoader(machine_dataset, batch_size=batch_size, shuffle=True)
    return dataloaders


def combine_dataloaders(dataloaders):
    all_sequences, all_targets, all_event_labels = [], [], []
    for dataloader in dataloaders.values():
        for sequences, targets, event_labels in dataloader:
            all_sequences.append(sequences)
            all_targets.append(targets)
            all_event_labels.append(event_labels)
    combined_dataset = TensorDataset(
        torch.cat(all_sequences, dim=0),
        torch.cat(all_targets, dim=0),
        torch.cat(all_event_labels, dim=0)
    )
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)


# 학습 함수
def train_model(model, dataloader, criterion, optimizer, num_epochs, device, metrics_save_path):
    # 디렉터리 생성
    metrics_dir = os.path.dirname(metrics_save_path)
    if metrics_dir and not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    # CSV 파일 초기화
    with open(metrics_save_path, "w") as f:
        f.write("epoch,loss,mse,rmse,r2\n")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        all_targets, all_reg_preds = [], []

        for sequences, targets, event_labels in dataloader:
            sequences, targets = sequences.to(device), targets.to(device)
            event_labels = event_labels.to(device)

            optimizer.zero_grad()
            reg_output = model(sequences, event_labels)
            loss = criterion(reg_output, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 성능 지표 계산
            all_targets.append(targets.cpu().detach().numpy())
            all_reg_preds.append(reg_output.cpu().detach().numpy())

        # Flatten targets and predictions
        all_targets = np.concatenate(all_targets, axis=0)
        all_reg_preds = np.concatenate(all_reg_preds, axis=0)

        # MSE, RMSE, R² 계산
        mse = mean_squared_error(all_targets, all_reg_preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets, all_reg_preds)

        # Epoch별 성능 지표 계산
        avg_loss = total_loss / len(dataloader)

        # 성능 지표 업데이트
        epoch_metrics["epoch"].append(epoch + 1)
        epoch_metrics["loss"].append(avg_loss)
        epoch_metrics["mse"].append(mse)
        epoch_metrics["rmse"].append(rmse)
        epoch_metrics["r2"].append(r2)

        # CSV 파일에 성능 지표 저장
        with open(metrics_save_path, "a") as f:
            f.write(f"{epoch + 1},{avg_loss:.4f},{mse:.4f},{rmse:.4f},{r2:.4f}\n")

        # 출력
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")


if __name__ == "__main__":
    print("모델 학습 시작")
    # 주요 설정
    sequence_length = 24
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.0005
    hidden_size = 100
    num_layers = 1
    features = ['average_usage_cpus', 'average_usage_memory', 'maximum_usage_cpus', 'maximum_usage_memory']
    input_size = len(features)
    output_size = len(features)
    event_embedding_dim = 3

    # 데이터 파일 경로
    file_path = '../../data/google_traces_v3/output_data_sampled2.csv'
    metrics_save_path = '../../data/metrics/metrics2_1.csv'
    model_save_path = './models/trained_lstm_model.pth'

    # 데이터 로드 및 전처리
    data = pd.read_csv(file_path)
    data = data[data['machine_id'] != -1]
    data['event_type_idx'] = pd.Categorical(data['event_type']).codes
    num_event_types = data['event_type_idx'].nunique()

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 준비
    dataloaders = prepare_dataloaders(data, sequence_length, batch_size, features)
    combined_dataloader = combine_dataloaders(dataloaders)

    # 모델 초기화
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        num_event_types=num_event_types,
        event_embedding_dim=event_embedding_dim
    ).to(device)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 모델 학습
    train_model(
        model=model,
        dataloader=combined_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        metrics_save_path=metrics_save_path
    )

    # 학습 완료 메시지
    print("모델 학습 완료")
    torch.save(model.state_dict(), model_save_path)
    print(f"모델이 저장되었습니다: {model_save_path}")