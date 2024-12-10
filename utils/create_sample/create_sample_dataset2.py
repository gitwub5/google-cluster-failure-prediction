import pandas as pd
import numpy as np

# 데이터 파일 경로
input_file = '../../data/google_traces_v3/instance_usage_preprocessed.csv'
train_output_file = '../../data/google_traces_v3/train_data.csv'
test_output_file = '../../data/google_traces_v3/test_data.csv'

# 학습 및 테스트 데이터에 사용할 머신 수
train_machine_count = 100
test_machine_count = 2

# 랜덤 시드 설정
np.random.seed(42)

# CSV 파일 읽기
df = pd.read_csv(input_file)

# `start_time`으로 정렬
df = df.sort_values(by=['machine_id', 'start_time'])

# 고유한 machine_id 추출
unique_machines = df['machine_id'].unique()

# 무작위로 섞기
np.random.shuffle(unique_machines)

# train과 test에 사용할 machine_id 분리
train_machines = unique_machines[:train_machine_count]
test_machines = unique_machines[train_machine_count:train_machine_count + test_machine_count]

# 데이터셋 분리
train_data = df[df['machine_id'].isin(train_machines)]
test_data = df[df['machine_id'].isin(test_machines)]

# CSV로 저장
train_data.to_csv(train_output_file, index=False)
test_data.to_csv(test_output_file, index=False)

# 결과 출력
print(f"Train dataset created with {len(train_data)} rows from {train_machine_count} machines.")
print(f"Test dataset created with {len(test_data)} rows from {test_machine_count} machines.")