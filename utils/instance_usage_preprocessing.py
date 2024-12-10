import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# CSV 파일 경로 설정
input_file = '../data/google_traces_v3/instance_usage_data.csv'
output_file = '../data/google_traces_v3/preprocessed_instance_usage.csv'

# CSV 파일 읽기
df = pd.read_csv(input_file)

# <start_time 기준 정렬>
df = df.sort_values(by=["machine_id", "start_time"]).reset_index(drop=True)
print(f"start_time 기준 정렬 성공")

# <결측값 처리 & 이상치 처리>
df = df.replace([np.inf, -np.inf], np.nan).dropna()
columns_to_check = ['average_usage_cpus', 'average_usage_memory', 'maximum_usage_cpus', 'maximum_usage_memory']
df = df[(df[columns_to_check] != 0).all(axis=1)]
if df.empty:
    raise ValueError("No data remaining after cleaning. Check input data or cleaning criteria.")
print(f"결측값 처리 & 이상치 처리 성공")

# <시간 구간으로 데이터를 집계>
interval = 300_000_000  # 5분 간격 (마이크로초 단위)
start_time = df['start_time'].min()
end_time = df['end_time'].max()

# 새로운 데이터를 저장할 리스트 초기화
aggregated_data = []

# 머신 ID별로 그룹화
for machine_id, group in df.groupby("machine_id"):
    current_time = start_time

    while current_time < end_time:
        next_time = current_time + interval
        interval_data = group[(group['start_time'] < next_time) & (group['end_time'] > current_time)]

        if interval_data.empty:
            current_time = next_time
            continue

        total_cpu_usage = 0
        total_memory_usage = 0
        for _, row in interval_data.iterrows():
            effective_start = max(row['start_time'], current_time)
            effective_end = min(row['end_time'], next_time)
            duration = effective_end - effective_start
            total_cpu_usage += row['average_usage_cpus'] * duration
            total_memory_usage += row['average_usage_memory'] * duration

        max_cpus = interval_data['maximum_usage_cpus'].max()
        max_memory = interval_data['maximum_usage_memory'].max()

        aggregated_data.append({
            'machine_id': machine_id,
            'start_time': current_time,
            'end_time': next_time,
            'average_usage_cpu': total_cpu_usage / interval,
            'average_usage_memory': total_memory_usage / interval,
            'maximum_usage_cpu': max_cpus,
            'maximum_usage_memory': max_memory
        })

        current_time = next_time

if aggregated_data:
    print("Sample aggregated data:")
    print(pd.DataFrame(aggregated_data).head())
else:
    print("No aggregated data was generated. Check input data and processing steps.")

# 전처리된 DataFrame 저장
aggregated_df = pd.DataFrame(aggregated_data)
aggregated_df.to_csv(output_file, index=False)
print(f"전처리 완료 및 파일 저장됨 -> {output_file}")