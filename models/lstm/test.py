import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from multi_time_series import SlidingWindowDataset, MultiTimeSeriesModel

print("Starting bigquery_test.py execution...")  # 맨 위에 추가

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_csv_path = '/Users/gwshin/Desktop/2024_khu/Capstone_Design/lstm-transformer/data/google_traces_v3/test_dataset.csv'
model_path = './models/multi_time_series_model.pth'


# 테스트 데이터셋 준비
window_size = 10
test_dataset = SlidingWindowDataset(test_csv_path, window_size=window_size)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

test_dataset.data['machine_id_idx'] = test_dataset.data['machine_id'].map(lambda x: x % 100)

# 모델 파라미터 설정 (학습 당시와 동일하게 설정)
input_size = 10
hidden_size = 64
num_layers = 1
output_size = 1
num_machines = 100
num_event_types = test_dataset.data['event_type'].nunique()
embedding_dim_machine = 8
embedding_dim_event = 2

# 모델 초기화 및 로드
model = MultiTimeSeriesModel(
    input_size, hidden_size, num_layers, output_size,
    num_machines=100,  # 저장된 모델의 설정에 맞춤
    num_event_types=test_dataset.data['event_type'].nunique(),
    embedding_dim_machine=embedding_dim_machine,
    embedding_dim_event=embedding_dim_event
).to(device)

# 학습된 모델 로드
try:
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 테스트 데이터로 예측 및 평가
all_preds = []
all_labels = []

with torch.no_grad():
    for sequences, machine_ids, event_types, labels in test_loader:
        sequences = sequences.to(device)
        machine_ids = machine_ids.to(device)
        event_types = event_types.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(sequences, machine_ids, event_types)
        preds = (torch.sigmoid(outputs.squeeze()) > 0.5).int()

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

# 평가 지표 계산
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=1)
recall = recall_score(all_labels, all_preds, zero_division=1)
f1 = f1_score(all_labels, all_preds, zero_division=1)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")