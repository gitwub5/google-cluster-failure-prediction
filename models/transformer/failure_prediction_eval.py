import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from failure_prediction import SequenceDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Transformer 모델 정의 (기존과 동일)
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

# 예측 및 CSV 저장 함수
def predict_and_save_results(model, dataloader, device, output_dir):
    model.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    grouped_results = {}
    with torch.no_grad():
        for sequences, event_labels, targets, machine_ids in dataloader:
            sequences, event_labels, targets = sequences.to(device), event_labels.to(device), targets.to(device)
            outputs = model(sequences, event_labels)
            probabilities = torch.sigmoid(outputs.squeeze()).cpu().numpy()
            actuals = targets.cpu().numpy()
            machine_ids = machine_ids.cpu().numpy()

            for i, machine_id in enumerate(machine_ids):
                grouped_results.setdefault(machine_id, {"Predicted": [], "Actual": []})
                grouped_results[machine_id]["Predicted"].append(probabilities[i])
                grouped_results[machine_id]["Actual"].append(actuals[i])

    # Save CSV and generate zplots2 for each machine
    for machine_id, results in grouped_results.items():
        # Convert results to DataFrame
        machine_results = pd.DataFrame({
            "Time Step": range(len(results["Predicted"])),
            "Predicted Probability": results["Predicted"],
            "Actual": results["Actual"]
        })

        # Save to CSV
        csv_path = os.path.join(output_dir, f"machine_{machine_id}_results.csv")
        machine_results.to_csv(csv_path, index=False)

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(machine_results["Time Step"], machine_results["Predicted Probability"], label="Predicted Probability", linestyle="--")
        plt.plot(machine_results["Time Step"], machine_results["Actual"], label="Actual", linestyle="-")
        plt.title(f"Machine ID: {machine_id}")
        plt.xlabel("Time Step")
        plt.ylabel("Probability / Actual")
        plt.legend()
        plot_path = os.path.join(output_dir, f"machine_{machine_id}_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Results saved for Machine ID {machine_id} as {csv_path} and plot as {plot_path}")

# 실행
if __name__ == "__main__":
    # 모델 로드
    model_path = './models/transformer_failure_model.pth'
    input_size = 4
    embedding_dim = 16
    num_heads = 4
    num_layers = 2
    output_size = 1
    sequence_length = 24
    features = ['average_usage_cpus', 'average_usage_memory', 'maximum_usage_cpus', 'maximum_usage_memory']
    target_col = 'Failed'

    # 테스트 데이터 로드
    test_file_path = '../../data/google_traces_v3/test_data.csv'
    test_data = pd.read_csv(test_file_path)
    test_data['event_type_idx'] = pd.Categorical(test_data['event_type']).codes
    # print("Test event_type_idx:", test_data['event_type_idx'].unique())

    num_event_types = test_data['event_type_idx'].nunique()

    # 모델 정의
    model = TransformerModel(input_size, embedding_dim, num_heads, num_layers, output_size, num_event_types)

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # 모델 로드
    if device.type == 'cpu':
        print("Loading model on CPU...")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print("Loading model on GPU...")
        model.load_state_dict(torch.load(model_path))

    # 디바이스로 모델 이동 및 평가 모드 설정
    model.to(device)
    model.eval()  # 평가 모드로 전환
    print(f"Model loaded from {model_path} and set to eval mode.")

    # 테스트 데이터셋 및 데이터로더 준비
    test_dataset = SequenceDataset(test_data, sequence_length, features, target_col=target_col)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 결과 저장 디렉터리
    output_dir = "../../data/results/failure"
    predict_and_save_results(model, test_dataloader, device, output_dir)