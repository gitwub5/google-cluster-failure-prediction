import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

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
    sequence_length = 50
    features = ['average_usage_cpus', 'average_usage_memory', 'maximum_usage_cpus', 'maximum_usage_memory']
    input_size = len(features)
    output_size = len(features)
    hidden_size = 100
    num_layers = 1
    event_embedding_dim = 3

    # 새로운 데이터 파일 경로
    predict_file_path = 'new_machine_data.csv'
    predict_data = pd.read_csv(predict_file_path)
    predict_data = predict_data[predict_data['machine_id'] != -1]
    predict_data['event_type_idx'] = pd.Categorical(predict_data['event_type']).codes
    num_event_types = predict_data['event_type_idx'].nunique()

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DataLoader 준비
    predict_dataloaders = prepare_predict_dataloader(predict_data, sequence_length, features)

    # 학습된 모델 로드
    model_path = "performance_predict_model.pth"
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, num_event_types, event_embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")

    # 성능 예측 및 평가
    predictions, actuals = predict_and_evaluate(model, predict_dataloaders, device)

    # 결과 저장
    result_df = pd.DataFrame({
        "Predicted": [list(p) for p in predictions],
        "Actual": [list(a) for a in actuals]
    })
    result_df.to_csv("prediction_results.csv", index=False)
    print("Predictions saved to prediction_results.csv")