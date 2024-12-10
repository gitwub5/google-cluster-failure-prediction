import pandas as pd
import numpy as np

train_machine_count = 20
test_machine_count = 10  # 학습 머신의 20%

# 데이터 파일 경로
input_file = '../../data/google_traces_v3/preprocessed_data.csv'
train_output_file = '../../data/google_traces_v3/train_data.csv'
test_output_file = '../../data/google_traces_v3/test_data.csv'

# CSV 파일 읽기
df = pd.read_csv(input_file)

# 각 machine_id별 데이터 개수 계산
machine_data_counts = df['machine_id'].value_counts()

# 100개 이상의 데이터를 가진 machine_id 필터링
valid_machine_ids = machine_data_counts[machine_data_counts >= 100].index

# 학습용으로 사용할 200개 머신 선택
if len(valid_machine_ids) < train_machine_count:
    raise ValueError(f"Not enough valid machines for training. Found {len(valid_machine_ids)}, need {train_machine_count}.")

train_machine_ids = np.random.choice(valid_machine_ids, size=train_machine_count, replace=False)

# 학습 데이터 필터링
train_data = df[df['machine_id'].isin(train_machine_ids)]

# 학습에 사용된 머신 ID를 제외한 나머지 머신 ID로 테스트 데이터 생성
remaining_machine_ids = valid_machine_ids[~valid_machine_ids.isin(train_machine_ids)]

# 테스트 머신의 개수 제한
if len(remaining_machine_ids) < test_machine_count:
    raise ValueError(f"Not enough remaining machines for testing. Found {len(remaining_machine_ids)}, need {test_machine_count}.")

test_machine_ids = np.random.choice(remaining_machine_ids, size=test_machine_count, replace=False)

# 테스트 데이터 필터링
test_data = df[df['machine_id'].isin(test_machine_ids)]

# 데이터 저장
train_data.to_csv(train_output_file, index=False)
test_data.to_csv(test_output_file, index=False)

print(f"Train dataset created with {len(train_data)} rows from {train_machine_count} machines.")
print(f"Test dataset created with {len(test_data)} rows from {test_machine_count} machines.")