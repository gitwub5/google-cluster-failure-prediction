import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 성능 지표 저장용 딕셔너리 선언
epoch_metrics = {
    "epoch": [],
    "loss": [],
    "mse": [],
    "rmse": [],
    "r2": [],
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": []
}


# 데이터셋 클래스 정의
class SequenceDataset(Dataset):
    def __init__(self, data, sequence_length, features):
        self.data = data
        self.sequence_length = sequence_length
        self.features = features
        self.machine_sequences = {} # machine_id별 시퀀스 저장
        self.prepare_data()

    def prepare_data(self):
        grouped = self.data.groupby('machine_id') # machine_id별 데이터 그룹화

        for machine_id, group in grouped:
            values = group[self.features].values # 여러 특성의 값 추출
            failed = group['Failed'].values
            event_types = group['event_type_idx'].values

            sequences, targets, failed_labels, event_labels = [], [], [], []
            for i in range(len(values) - self.sequence_length):
                sequences.append(values[i:i + self.sequence_length])
                targets.append(values[i + self.sequence_length])
                failed_labels.append(failed[i + self.sequence_length])
                event_labels.append(event_types[i + self.sequence_length])

            self.machine_sequences[machine_id] = {
                'sequences': sequences,
                'targets': targets,
                'failed_labels': failed_labels,
                'event_labels': event_labels
            }

    def get_machine_dataset(self, machine_id):
        if machine_id not in self.machine_sequences:
            raise ValueError(f"Machine ID {machine_id} not found in dataset.")
        data = self.machine_sequences[machine_id]

        # 각 데이터 변환
        sequences = torch.tensor(np.array(data['sequences']), dtype=torch.float32)
        targets = torch.tensor(np.array(data['targets']), dtype=torch.float32)
        failed_labels = torch.tensor(np.array(data['failed_labels']), dtype=torch.float32)
        event_labels = torch.tensor(np.array(data['event_labels']), dtype=torch.long)

        # 데이터 길이 확인
        assert len(sequences) == len(targets) == len(failed_labels) == len(event_labels), \
            "Lengths of sequences, targets, failed_labels, and event_labels must match."

        return TensorDataset(sequences, targets, failed_labels, event_labels)


# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_event_types, event_embedding_dim):
        """
        LSTM 기반 다중 출력 모델
        :param input_size: 입력 특징의 크기 (number of features)
        :param hidden_size: LSTM 히든 레이어 크기
        :param num_layers: LSTM 레이어 수
        :param output_size: 회귀 출력 크기 (예측할 연속형 값의 개수)
        :param num_event_types: 이벤트 종류의 개수 (event_type의 고유 값 개수)
        :param event_embedding_dim: 이벤트 임베딩 차원
        """
        super(LSTMModel, self).__init__()

        # Event Type Embedding Layer
        # event_type (카테고리 데이터) 정보를 실수 벡터로 임베딩
        self.event_embedding = nn.Embedding(num_event_types, event_embedding_dim)

        # LSTM Layer
        # 입력 크기는 (특징 크기 + 이벤트 임베딩 크기)
        self.lstm = nn.LSTM(input_size + event_embedding_dim, hidden_size, num_layers, batch_first=True)

        # Fully Connected Layer for Regression
        # LSTM의 최종 히든 상태를 받아 연속형 값을 예측
        self.fc_regression = nn.Linear(hidden_size, output_size)

        # Fully Connected Layer for Classification
        # LSTM의 최종 히든 상태를 받아 이진 분류(Failed 여부)를 예측
        self.fc_classification = nn.Linear(hidden_size, 1)

    def forward(self, x, event_type_idx):
        """
        모델의 Forward Pass
        :param x: 입력 시계열 데이터 (batch_size, sequence_length, input_size)
        :param event_type_idx: 이벤트 타입 인덱스 (batch_size,)
        :return: (회귀 출력, 분류 출력)
        """
        # Event Type Embedding
        # event_type 인덱스를 임베딩 벡터로 변환
        event_embedding = self.event_embedding(event_type_idx)  # (batch_size, embedding_dim)

        # 임베딩 벡터를 각 시계열 타임스텝에 적용
        # (batch_size, sequence_length, embedding_dim)
        event_embedding = event_embedding.unsqueeze(1).repeat(1, x.size(1), 1)

        # 입력 데이터와 이벤트 임베딩을 결합
        # 결합된 입력 크기: (batch_size, sequence_length, input_size + embedding_dim)
        x = torch.cat((x, event_embedding), dim=2)

        # LSTM Layer Forward Pass
        # LSTM은 최종 히든 상태(hidden)와 셀 상태(cell)를 반환
        _, (hidden, _) = self.lstm(x)

        # 최종 LSTM 레이어의 마지막 히든 상태를 사용
        # hidden의 크기: (num_layers, batch_size, hidden_size)
        hidden = hidden[-1]  # (batch_size, hidden_size)

        # Regression Output (연속형 변수 예측)
        regression_output = self.fc_regression(hidden)  # (batch_size, output_size)

        # Classification Output (Failed 여부 예측)
        classification_output = self.fc_classification(hidden)  # (batch_size, 1)

        return regression_output, classification_output

def prepare_dataloaders(data, sequence_length, batch_size, features):
    dataset = SequenceDataset(data, sequence_length, features)
    dataloaders = {}
    for machine_id in dataset.machine_sequences.keys():
        if len(dataset.machine_sequences[machine_id]['sequences']) == 0:
            continue
        machine_dataset = dataset.get_machine_dataset(machine_id)
        dataloaders[machine_id] = DataLoader(machine_dataset, batch_size=batch_size, shuffle=True)
        # print(f"Machine ID {machine_id}, Total Sequences: {len(machine_dataset)}")
    return dataloaders

def combine_dataloaders(dataloaders):
    all_sequences, all_targets, all_failed_labels, all_event_labels = [], [], [], []
    for dataloader in dataloaders.values():
        for sequences, targets, failed_labels, event_labels in dataloader:
            all_sequences.append(sequences)
            all_targets.append(targets)
            all_failed_labels.append(failed_labels)
            all_event_labels.append(event_labels)
    combined_dataset = TensorDataset(
        torch.cat(all_sequences, dim=0),
        torch.cat(all_targets, dim=0),
        torch.cat(all_failed_labels, dim=0),
        torch.cat(all_event_labels, dim=0)
    )
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

# 학습 함수
def train_model(model, dataloader, criterion_reg, criterion_cls, optimizer, num_epochs, device, metrics_save_path):
    # 디렉터리 생성 (폴더가 없는 경우)
    metrics_dir = os.path.dirname(metrics_save_path)
    if metrics_dir and not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    # CSV 파일 초기화 (헤더 작성)
    if not os.path.exists(metrics_save_path):
        with open(metrics_save_path, "w") as f:
            f.write("epoch,loss,mse,rmse,r2,accuracy,precision,recall,f1_score\n")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        all_targets, all_reg_preds = [], []  # 회귀 지표 계산용
        all_cls_preds, all_failed_labels = [], []  # 분류 지표 계산용

        for sequences, targets, failed_labels, event_labels in dataloader:
            sequences, targets = sequences.to(device), targets.to(device)
            failed_labels, event_labels = failed_labels.to(device), event_labels.to(device)

            optimizer.zero_grad()
            reg_output, cls_output = model(sequences, event_labels)
            reg_loss = criterion_reg(reg_output, targets)
            cls_loss = criterion_cls(cls_output.squeeze(), failed_labels)
            loss = reg_loss + cls_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 회귀 성능 지표 계산
            all_targets.append(targets.cpu().detach().numpy())
            all_reg_preds.append(reg_output.cpu().detach().numpy())

            # 분류 성능 지표 계산
            preds = (torch.sigmoid(cls_output.squeeze()) > 0.5).int()
            all_cls_preds.append(preds.cpu().numpy())
            all_failed_labels.append(failed_labels.cpu().numpy())

        # Flatten targets and predictions for metric calculation
        all_targets = np.concatenate(all_targets, axis=0)
        all_reg_preds = np.concatenate(all_reg_preds, axis=0)
        all_cls_preds = np.concatenate(all_cls_preds, axis=0)
        all_failed_labels = np.concatenate(all_failed_labels, axis=0)

        # MSE, RMSE, R² 계산
        mse = mean_squared_error(all_targets, all_reg_preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets, all_reg_preds)

        # Accuracy, Precision, Recall, F1 계산
        accuracy = accuracy_score(all_failed_labels, all_cls_preds)
        precision = precision_score(all_failed_labels, all_cls_preds, zero_division=1)
        recall = recall_score(all_failed_labels, all_cls_preds, zero_division=1)
        f1 = f1_score(all_failed_labels, all_cls_preds, zero_division=1)

        # Epoch별 성능 지표 계산
        avg_loss = total_loss / len(dataloader)

        # 성능 지표 업데이트
        epoch_metrics["epoch"].append(epoch + 1)
        epoch_metrics["loss"].append(avg_loss)
        epoch_metrics["mse"].append(mse)
        epoch_metrics["rmse"].append(rmse)
        epoch_metrics["r2"].append(r2)
        epoch_metrics["accuracy"].append(accuracy)
        epoch_metrics["precision"].append(precision)
        epoch_metrics["recall"].append(recall)
        epoch_metrics["f1_score"].append(f1)

        # CSV 파일에 성능 지표 저장
        with open(metrics_save_path, "a") as f:
            f.write(
                f"{epoch + 1},{avg_loss:.4f},{mse:.4f},{rmse:.4f},{r2:.4f},{accuracy:.4f},{precision:.4f},{recall:.4f},{f1:.4f}\n")

        # 출력
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, "
            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

if __name__ == "__main__":
    # 주요 설정
    sequence_length = 100  # 슬라이딩 윈도우 크기
    batch_size = 32  # 배치 크기
    num_epochs = 50  # 학습 반복 수
    learning_rate = 0.0001  # 학습률
    hidden_size = 100  # LSTM 히든 레이어 크기
    num_layers = 2  # LSTM 레이어 수
    features = ['average_usage_cpus', 'average_usage_memory', 'maximum_usage_cpus', 'maximum_usage_memory']  # 입력 특징
    input_size = len(features)  # 입력 차원 크기
    output_size = len(features)  # 회귀 출력 차원 크기
    event_embedding_dim = 3  # event_type 임베딩 차원

    # 데이터 파일 경로
    file_path = '../../data/google_traces_v3/output_data_sampled.csv'
    metrics_save_path = '../../data/metrics/metrics.csv'
    model_save_path = './models/trained_lstm_model.pth'  # 모델 저장 경로

    # 데이터 로드 및 전처리
    data = pd.read_csv(file_path)  # CSV 파일 로드
    data = data[data['machine_id'] != -1]  # 유효한 machine_id만 필터링

    # event_type 값을 0부터 시작하는 연속된 값으로 매핑
    data['event_type_idx'] = pd.Categorical(data['event_type']).codes
    num_event_types = data['event_type_idx'].nunique()  # 고유 event_type 수 계산
    # print(f"Mapped event_type values: {data['event_type_idx'].unique()}")
    # print(f"Number of event types: {num_event_types}")

    # 디바이스 설정 (GPU가 가능하면 GPU 사용, 그렇지 않으면 CPU 사용)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 준비 (machine_id별 슬라이딩 윈도우 생성)
    dataloaders = prepare_dataloaders(data, sequence_length, batch_size, features)

    # 모든 머신 데이터를 하나로 결합
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

    # 손실 함수 정의
    criterion_reg = nn.MSELoss()  # 회귀 손실 함수
    criterion_cls = nn.BCEWithLogitsLoss()  # 분류 손실 함수

    # 옵티마이저 설정
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 모델 학습
    train_model(
        model=model,
        dataloader=combined_dataloader,
        criterion_reg=criterion_reg,
        criterion_cls=criterion_cls,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        metrics_save_path=metrics_save_path
    )

    # 학습 완료 메시지
    print("모델 학습 완료")

    # 학습된 모델 저장
    torch.save(model.state_dict(), model_save_path)
    print(f"모델이 저장되었습니다: {model_save_path}")