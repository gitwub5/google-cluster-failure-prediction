import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader, TensorDataset

# LSTM 모델 클래스 정의 (이전 코드와 동일)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_event_types, event_embedding_dim):
        super(LSTMModel, self).__init__()
        self.event_embedding = nn.Embedding(num_event_types, event_embedding_dim)
        self.lstm = nn.LSTM(input_size + event_embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc_regression = nn.Linear(hidden_size, output_size)

    def forward(self, x, event_type_idx):
        event_embedding = self.event_embedding(event_type_idx).unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat((x, event_embedding), dim=2)
        _, (hidden, _) = self.lstm(x)
        regression_output = self.fc_regression(hidden[-1])
        return regression_output

# 데이터셋 클래스 정의 (이전 코드와 동일)
class SequenceDataset(Dataset):
    def __init__(self, data, sequence_length, features):
        self.data = data
        self.sequence_length = sequence_length
        self.features = features
        self.machine_sequences = {}
        self.prepare_data()

    def prepare_data(self):
        grouped = self.data.groupby('machine_id')
        for machine_id, group in grouped:
            values = group[self.features].values
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
        data = self.machine_sequences[machine_id]
        sequences = torch.tensor(np.array(data['sequences']), dtype=torch.float32)
        targets = torch.tensor(np.array(data['targets']), dtype=torch.float32)
        event_labels = torch.tensor(np.array(data['event_labels']), dtype=torch.long)
        return TensorDataset(sequences, targets, event_labels)

# 새로운 데이터 준비
def prepare_predict_dataloader(data, sequence_length, features):
    dataset = SequenceDataset(data, sequence_length, features)
    dataloaders = {}
    for machine_id in dataset.machine_sequences.keys():
        if len(dataset.machine_sequences[machine_id]['sequences']) == 0:
            continue
        machine_dataset = dataset.get_machine_dataset(machine_id)
        dataloaders[machine_id] = DataLoader(machine_dataset, batch_size=1, shuffle=False)
    return dataloaders

# 성능 예측 및 평가
def predict_and_evaluate(model, dataloaders, device):
    predictions, actuals = [], []
    for machine_id, dataloader in dataloaders.items():
        print(f"Predicting for Machine ID: {machine_id}")
        with torch.no_grad():
            for inputs, targets, event_labels in dataloader:
                inputs, event_labels = inputs.to(device), event_labels.to(device)
                outputs = model(inputs, event_labels)
                predictions.append(outputs.cpu().numpy())
                actuals.append(targets.numpy())
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    # 평가 지표 계산
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)

    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return predictions, actuals

if __name__ == "__main__":
    # 주요 설정
    sequence_length = 24
    features = ['average_usage_cpus', 'average_usage_memory', 'maximum_usage_cpus', 'maximum_usage_memory']
    input_size = len(features)
    output_size = len(features)
    hidden_size = 100
    num_layers = 1
    event_embedding_dim = 3

    # 새로운 데이터 파일 경로
    predict_file_path = '../../data/google_traces_v3/test_data.csv'
    predict_data = pd.read_csv(predict_file_path)
    predict_data = predict_data[predict_data['machine_id'] != -1]
    predict_data['event_type_idx'] = pd.Categorical(predict_data['event_type']).codes
    num_event_types = predict_data['event_type_idx'].nunique()

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DataLoader 준비
    predict_dataloaders = prepare_predict_dataloader(predict_data, sequence_length, features)

    # 학습된 모델 로드 및 평가 모드 설정
    model_path = "models/trained_lstm_model.pth"
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, num_event_types, event_embedding_dim)

    if device.type == 'cpu':
        print("Loading model on CPU...")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print("Loading model on GPU...")
        model.load_state_dict(torch.load(model_path))

    model.to(device)
    model.eval()  # 평가 모드로 전환
    print(f"Model loaded from {model_path} and set to eval mode.")

    # 결과 저장 디렉터리 및 파일 경로 설정
    output_dir = '../../data/results/instance_usage'
    output_file = 'prediction_results.csv'
    output_path = os.path.join(output_dir, output_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory {output_dir} created.")

    # 결과 저장을 위한 리스트 초기화
    results = []

    # 각 machine_id에 대해 예측 수행 및 결과 저장
    for machine_id, dataloader in predict_dataloaders.items():
        machine_predictions = []
        machine_actuals = []

        with torch.no_grad():
            for inputs, targets, event_labels in dataloader:
                inputs, event_labels = inputs.to(device), event_labels.to(device)
                outputs = model(inputs, event_labels)
                machine_predictions.append(outputs.cpu().numpy())
                machine_actuals.append(targets.numpy())

        machine_predictions = np.concatenate(machine_predictions, axis=0)
        machine_actuals = np.concatenate(machine_actuals, axis=0)

        # 머신 ID별 데이터 추가
        for pred, actual in zip(machine_predictions, machine_actuals):
            results.append({
                "Machine ID": machine_id,  # 머신 ID 추가
                "Predicted": list(pred),  # 예측값
                "Actual": list(actual)  # 실제값
            })

    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(results)

    # 결과 저장
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")