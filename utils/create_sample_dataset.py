import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
#
# # CSV 파일 읽기
# df = pd.read_csv('../data/google_traces_v3/output_data_preprocessed.csv')
#
# # 각 machine_id별 데이터 개수 계산
# machine_data_counts = df['machine_id'].value_counts()
#
# # 무작위로 1000개의 machine_id 선택
# random_machine_ids = np.random.choice(machine_data_counts.index, size=200, replace=False)
#
# # 선택된 machine_id에 해당하는 모든 데이터 필터링
# filtered_df = df[df['machine_id'].isin(random_machine_ids)]
#
# # event_time 기준으로 정렬
# filtered_df = filtered_df.sort_values(by='event_time')
#
# # 필터링된 데이터셋 저장
# filtered_df.to_csv('../data/google_traces_v3/output_data_sampled.csv', index=False)
#
# print(f"Filtered dataset created with {len(filtered_df)} rows from 1000 machines.")



# CSV 파일 읽기
df = pd.read_csv('../data/google_traces_v3/output_data_preprocessed2.csv')

# 각 machine_id별 데이터 개수 계산
machine_data_counts = df['machine_id'].value_counts()

# 200개 이상의 데이터를 가진 machine_id 필터링
valid_machine_ids = machine_data_counts[machine_data_counts >= 100].index

# 필터링된 machine_id에서 랜덤하게 10개 선택
random_machine_ids = np.random.choice(valid_machine_ids, size=1000, replace=False)

# 선택된 machine_id에 해당하는 모든 데이터 필터링
filtered_df = df[df['machine_id'].isin(random_machine_ids)]

# event_time 기준으로 정렬
filtered_df = filtered_df.sort_values(by='event_time')

# 필터링된 데이터셋 저장
filtered_df.to_csv('../data/google_traces_v3/output_data_sampled2.csv', index=False)

print(f"Filtered dataset created with {len(filtered_df)} rows.")