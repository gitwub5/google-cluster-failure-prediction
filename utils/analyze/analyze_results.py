import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta

# CSV 파일 읽기
result_file = '../../data/results/instance_usage/prediction_results.csv'
df = pd.read_csv(result_file)

# 문자열로 저장된 리스트를 실제 리스트로 변환
df['Predicted'] = df['Predicted'].apply(ast.literal_eval)
df['Actual'] = df['Actual'].apply(ast.literal_eval)

# 기준 시간 설정
base_time = pd.Timestamp("2019-05-01 00:00:00", tz="America/New_York")
# start_time을 마이크로초 단위에서 초 단위로 변환 후 기준 시간에 더하기
df['Start Time'] = base_time + pd.to_timedelta(df['Start Time'], unit='us')

# Machine ID별로 데이터 그룹화
grouped = df.groupby('Machine ID')

# 결과를 저장할 디렉터리
base_output_dir = '../../data/results/instance_usage/plots/'
os.makedirs(base_output_dir, exist_ok=True)

# 성능 지표 저장용 리스트
performance_metrics = []

# 각 Machine ID에 대해 그래프 생성
for machine_id, group in grouped:
    # 날짜별 데이터 분리
    group['Date'] = group['Start Time'].dt.date
    date_groups = group.groupby('Date')

    predicted = np.array(group['Predicted'].tolist())
    actual = np.array(group['Actual'].tolist())
    start_times = group['Start Time'].values  # Start Time 추가

    # 특성 개수 확인
    num_features = predicted.shape[1]

    # 머신별 디렉터리 생성
    machine_output_dir = os.path.join(base_output_dir, f'machine_{machine_id}')
    os.makedirs(machine_output_dir, exist_ok=True)

    for date, date_group in date_groups:
        date_predicted = np.array(date_group['Predicted'].tolist())
        date_actual = np.array(date_group['Actual'].tolist())
        date_start_times = date_group['Start Time'].values

        for i in range(num_features):
            # Feature별 예측값과 실제값 추출
            predicted_feature = date_predicted[:, i]
            actual_feature = date_actual[:, i]

            # MSE 및 R² 계산
            mse = mean_squared_error(actual_feature, predicted_feature)
            r2 = r2_score(actual_feature, predicted_feature)

            # 성능 지표 기록
            performance_metrics.append({
                'Machine ID': machine_id,
                'Feature': f'Feature {i + 1}',
                'MSE': mse,
                'R²': r2,
                'Start Date': date
            })

            # 그래프 생성
            plt.figure(figsize=(10, 6))
            plt.plot(date_start_times, predicted_feature, label='Predicted', linestyle='--', marker='o')
            plt.plot(date_start_times, actual_feature, label='Actual', linestyle='-', marker='x')
            plt.title(f'Machine ID: {machine_id} - Feature {i + 1} - Date: {date}\nMSE: {mse:.4f}, R²: {r2:.4f}')
            plt.xlabel('Start Time')
            plt.ylabel('Value')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()

            # 그래프 저장
            output_file = os.path.join(machine_output_dir, f'date_{date}_feature_{i + 1}.png')
            plt.savefig(output_file)
            plt.close()
            print(f"Plot saved for Machine ID {machine_id}, Date {date}, Feature {i + 1}: {output_file}")

# 성능 지표를 DataFrame으로 변환
metrics_df = pd.DataFrame(performance_metrics)

# 성능 지표 저장 경로
metrics_output_file = '../../data/results/instance_usage/performance_metrics.csv'
metrics_df.to_csv(metrics_output_file, index=False)
print(f"Performance metrics saved to {metrics_output_file}")