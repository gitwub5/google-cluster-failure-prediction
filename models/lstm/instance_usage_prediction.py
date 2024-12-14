import os
import time  # 시간 측정을 위한 모듈
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from early_stopping import EarlyStopping

# 성능 지표 저장용 딕셔너리 선언
epoch_metrics = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
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

            sequences, targets = [], []
            for i in range(len(values) - self.sequence_length):
                sequences.append(values[i:i + self.sequence_length])
                targets.append(values[i + self.sequence_length])

            self.machine_sequences[machine_id] = {
                'sequences': sequences,
                'targets': targets
            }

    def get_machine_dataset(self, machine_id):
        if machine_id not in self.machine_sequences:
            raise ValueError(f"Machine ID {machine_id} not found in dataset.")
        data = self.machine_sequences[machine_id]
        sequences = torch.tensor(np.array(data['sequences']), dtype=torch.float32)
        targets = torch.tensor(np.array(data['targets']), dtype=torch.float32)
        return TensorDataset(sequences, targets)


# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        LSTM 기반 성능 예측 모델 (event_type 임베딩 제거)
        """
        super(LSTMModel, self).__init__()
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully Connected Layer for Regression
        self.fc_regression = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        모델의 Forward Pass
        """
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
        dataloaders[machine_id] = DataLoader(machine_dataset, batch_size=batch_size, shuffle=False)
    return dataloaders


def combine_dataloaders(dataloaders):
    all_sequences, all_targets = [], []
    for machine_id, dataloader in dataloaders.items():
        for sequences, targets in dataloader:
            all_sequences.append(sequences)
            all_targets.append(targets)
    combined_dataset = TensorDataset(
        torch.cat(all_sequences, dim=0),
        torch.cat(all_targets, dim=0)
    )
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)


# 학습 함수
def train_model(model, train_dataloader, criterion, optimizer, device):
    start_time = time.time()  # Epoch 시작 시간
    model.train()
    total_loss = 0.0
    all_targets, all_reg_preds = [], []

    for sequences, targets in train_dataloader:
        sequences, targets = sequences.to(device), targets.to(device)

        optimizer.zero_grad()  # 기울기 초기화
        reg_output = model(sequences)  # 모델 출력 계산
        loss = criterion(reg_output, targets)  # 손실 계산
        loss.backward()  # 역전파
        optimizer.step()  # 가중치 업데이트
        total_loss += loss.item()

    train_time = time.time() - start_time
    avg_loss = total_loss / len(train_dataloader)  # 평균 손실 계산
    return avg_loss, train_time

def evaluate_model(model, val_dataloader, criterion, device):
    model.eval()  # 평가 모드 설정
    total_loss = 0.0
    all_targets, all_reg_preds = [], []

    with torch.no_grad():  # 평가 시에는 기울기 계산 비활성화
        for sequences, targets in val_dataloader:
            sequences, targets = sequences.to(device), targets.to(device)
            reg_output = model(sequences)
            loss = criterion(reg_output, targets)
            total_loss += loss.item()

            # 예측값 및 실제값 저장
            all_targets.append(targets.cpu().numpy())
            all_reg_preds.append(reg_output.cpu().numpy())

    avg_loss = total_loss / len(val_dataloader)  # 평균 손실 계산
    all_targets = np.concatenate(all_targets, axis=0)
    all_reg_preds = np.concatenate(all_reg_preds, axis=0)

    # 성능 메트릭 계산
    mse = mean_squared_error(all_targets, all_reg_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_reg_preds)

    return avg_loss, mse, rmse, r2

if __name__ == "__main__":
    start_time = time.time()

    print("Step 1: 데이터 로드 및 전처리 시작")
    data_load_start = time.time()
    # 주요 설정
    sequence_length = 24
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.0005
    hidden_size = 128
    num_layers = 8
    features = ['average_usage_cpus', 'average_usage_memory', 'maximum_usage_cpus', 'maximum_usage_memory']
    input_size = len(features)
    output_size = len(features)

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터 파일 경로
    file_path = '../../data/google_traces_v3/train_data.csv'
    metrics_save_path = '../../data/metrics/metrics.csv'
    model_save_path = './models/trained_lstm_model.pth'

    # 데이터 로드 및 전처리
    data = pd.read_csv(file_path)
    data = data[data['machine_id'] != -1]
    print(f"Step 1 완료, 소요 시간: {time.time() - data_load_start:.2f}초")

    print("Step 2: 데이터셋 준비")
    dataloader_start = time.time()
    dataloaders = prepare_dataloaders(data, sequence_length, batch_size, features)
    combined_dataloader = combine_dataloaders(dataloaders)

    # Train-validation split
    dataset = combined_dataloader.dataset  # 데이터셋 가져오기
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(combined_dataloader.dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Step 2 완료, 소요 시간: {time.time() - dataloader_start:.2f}초")

    print("Step 3: 모델 초기화")
    model_init_start = time.time()
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
    ).to(device)
    print(f"Step 3 완료, 소요 시간: {time.time() - model_init_start:.2f}초")

    print("Step 4: 학습 시작")
    training_start = time.time()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # EarlyStopping 초기화
    early_stopping = EarlyStopping(patience=10, verbose=True, path=model_save_path)

    # CSV 파일 초기화
    with open(metrics_save_path, "w") as f:
        f.write("epoch,train_loss,val_loss,mse,rmse,r2\n")

    # 학습 및 평가
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_time = train_model(model, train_dataloader, criterion, optimizer, device)
        val_loss, mse, rmse, r2 = evaluate_model(model, val_dataloader, criterion, device)

        # 성능 지표 저장
        epoch_metrics["epoch"].append(epoch + 1)
        epoch_metrics["train_loss"].append(train_loss)
        epoch_metrics["val_loss"].append(val_loss)
        epoch_metrics["mse"].append(mse)
        epoch_metrics["rmse"].append(rmse)
        epoch_metrics["r2"].append(r2)

        with open(metrics_save_path, "a") as f:
            f.write(f"{epoch + 1},{train_loss:.4f},{val_loss:.4f},{mse:.4f},{rmse:.4f},{r2:.4f}\n")

        # 성능 지표 출력
        print(
            f"Train Loss: {train_loss:.4f}, "
            f"Validation Loss: {val_loss:.4f}, "
            f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, "
            f"Time: {train_time:.4f} "
        )

        # Early Stopping 적용
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"{epoch + 1} - Early stopping triggered.")
            break

    print(f"Step 4 완료, 학습 소요 시간: {time.time() - training_start:.2f}초")

    # 모델 저장
    print("Step 5: 모델 저장")
    model_save_start = time.time()
    torch.save(model.state_dict(), model_save_path)
    print(f"Step 5 완료, 모델 저장 완료. 소요 시간: {time.time() - model_save_start:.2f}초")

    print(f"전체 소요 시간: {time.time() - start_time:.2f}초")