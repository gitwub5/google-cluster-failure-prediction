import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# CSV 파일 읽기
result_file = '../data/results/prediction_results.csv'
df = pd.read_csv(result_file)

# 문자열로 저장된 리스트를 실제 리스트로 변환
df['Predicted'] = df['Predicted'].apply(ast.literal_eval)
df['Actual'] = df['Actual'].apply(ast.literal_eval)

# Machine ID별로 데이터 그룹화
grouped = df.groupby('Machine ID')

# 결과를 저장할 디렉터리
base_output_dir = '../data/results/plots/'
os.makedirs(base_output_dir, exist_ok=True)

# 성능 지표 저장용 리스트
performance_metrics = []

# 각 Machine ID에 대해 그래프 생성
for machine_id, group in grouped:
    predicted = np.array(group['Predicted'].tolist())
    actual = np.array(group['Actual'].tolist())

    # 특성 개수 확인
    num_features = predicted.shape[1]

    # 머신별 디렉터리 생성
    machine_output_dir = os.path.join(base_output_dir, f'machine_{machine_id}')
    os.makedirs(machine_output_dir, exist_ok=True)

    for i in range(num_features):
        # Feature별 예측값과 실제값 추출
        predicted_feature = predicted[:, i]
        actual_feature = actual[:, i]

        # MSE 및 R² 계산
        mse = mean_squared_error(actual_feature, predicted_feature)
        r2 = r2_score(actual_feature, predicted_feature)

        # 성능 지표 기록
        performance_metrics.append({
            'Machine ID': machine_id,
            'Feature': f'Feature {i + 1}',
            'MSE': mse,
            'R²': r2
        })

        # 그래프 생성
        plt.figure(figsize=(10, 6))
        plt.plot(predicted_feature, label='Predicted', linestyle='--')
        plt.plot(actual_feature, label='Actual', linestyle='-')
        plt.title(f'Machine ID: {machine_id} - Feature {i + 1}\nMSE: {mse:.4f}, R²: {r2:.4f}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()

        # 그래프 저장
        output_file = os.path.join(machine_output_dir, f'feature_{i + 1}.png')
        plt.savefig(output_file)
        plt.close()
        print(f"Plot saved for Machine ID {machine_id}, Feature {i + 1}: {output_file}")

# 성능 지표를 DataFrame으로 변환
metrics_df = pd.DataFrame(performance_metrics)

# 성능 지표 저장 경로
metrics_output_file = '../data/results/performance_metrics.csv'
metrics_df.to_csv(metrics_output_file, index=False)
print(f"Performance metrics saved to {metrics_output_file}")