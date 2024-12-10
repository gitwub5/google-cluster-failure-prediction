import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader, TensorDataset

# LSTM 모델 클래스 정의 (이전 코드와 동일)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_regression = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        regression_output = self.fc_regression(hidden[-1])
        return regression_output

# 데이터셋 클래스 정의
class SequenceDataset(Dataset):
    def __init__(self, data, sequence_length, features):
        self.data = data
        self.sequence_length = sequence_length
        self.features = features
        self.machine_sequences = {}
        self.start_times = {}  # start_time 저장
        self.prepare_data()

    def prepare_data(self):
        grouped = self.data.groupby('machine_id')
        for machine_id, group in grouped:
            values = group[self.features].values
            sequences, targets = [], []
            start_times = []  # start_time 저장용 리스트
            for i in range(len(values) - self.sequence_length):
                sequences.append(values[i:i + self.sequence_length])
                targets.append(values[i + self.sequence_length])
                start_times.append(group.iloc[i + self.sequence_length]['start_time'])  # start_time 추가
            self.machine_sequences[machine_id] = {
                'sequences': sequences,
                'targets': targets,
            }
            self.start_times[machine_id] = start_times  # start_time 저장

    def get_machine_dataset(self, machine_id):
        data = self.machine_sequences[machine_id]
        sequences = torch.tensor(np.array(data['sequences']), dtype=torch.float32)
        targets = torch.tensor(np.array(data['targets']), dtype=torch.float32)
        start_times = self.start_times[machine_id]
        return TensorDataset(sequences, targets), start_times


# DataLoader 준비
def prepare_predict_dataloader(data, sequence_length, features):
    dataset = SequenceDataset(data, sequence_length, features)
    dataloaders = {}
    for machine_id in dataset.machine_sequences.keys():
        if len(dataset.machine_sequences[machine_id]['sequences']) == 0:
            continue
        machine_dataset, start_times = dataset.get_machine_dataset(machine_id)
        dataloaders[machine_id] = (DataLoader(machine_dataset, batch_size=1, shuffle=False), start_times)
    return dataloaders

# 성능 예측 및 평가
def predict_and_evaluate_with_time(model, dataloaders, device):
    results = []
    for machine_id, (dataloader, start_times) in dataloaders.items():
        print(f"Predicting for Machine ID: {machine_id}")
        machine_predictions, machine_actuals, machine_start_times = [], [], []
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(device)
                outputs = model(inputs)
                machine_predictions.append(outputs.cpu().numpy())
                machine_actuals.append(targets.numpy())
                machine_start_times.append(start_times[idx])  # start_time 추가
        for pred, actual, start_time in zip(machine_predictions, machine_actuals, machine_start_times):
            results.append({
                "Machine ID": machine_id,
                "Start Time": start_time,
                "Predicted": list(pred.flatten()),
                "Actual": list(actual.flatten())
            })
    return results


if __name__ == "__main__":
    # 주요 설정
    sequence_length = 24
    features = ['average_usage_cpus', 'average_usage_memory', 'maximum_usage_cpus', 'maximum_usage_memory']
    input_size = len(features)
    output_size = len(features)
    hidden_size = 128
    num_layers = 4

    # 데이터 파일 경로
    predict_file_path = '../../data/google_traces_v3/test_data.csv'
    predict_data = pd.read_csv(predict_file_path)
    predict_data = predict_data[predict_data['machine_id'] != -1]

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DataLoader 준비
    predict_dataloaders = prepare_predict_dataloader(predict_data, sequence_length, features)

    # 학습된 모델 로드 및 평가 모드 설정
    model_path = "models/trained_lstm_model.pth"
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

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

    # 예측 및 결과 저장
    results = predict_and_evaluate_with_time(model, predict_dataloaders, device)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")