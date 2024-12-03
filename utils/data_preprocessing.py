import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# CSV 파일 경로 설정
input_file = '../data/google_traces_v3/output_data.csv'
output_file = '../data/google_traces_v3/output_data_preprocessed2.csv'

# CSV 파일 읽기
df = pd.read_csv(input_file)

# <결측값 처리 & 이상치 처리>
# NaN  or NULL, Inf 값 제거
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# 특정 열 중 하나라도 0인 행 제거
columns_to_check = [
    'average_usage_cpus', 'average_usage_memory',
    'maximum_usage_cpus', 'maximum_usage_memory'
]

df = df[(df[columns_to_check] != 0).all(axis=1)]

print(f"결측값 처리 & 이상치 처리 성공")


# <Min-Max 정규화>
columns_to_normalize = [
    'average_usage_cpus', 'average_usage_memory',
    'maximum_usage_cpus', 'maximum_usage_memory'
]

scaler = MinMaxScaler()

# 스케일링 적용 및 데이터프레임에 반영
scaled_data = scaler.fit_transform(df[columns_to_normalize])
df[columns_to_normalize] = scaled_data

print(f"Min-Max 정규화 성공")

# Failed 열 추가 (event_type이 5인 경우 1, 아니면 0)
df['Failed'] = df['event_type'].apply(lambda x: 1 if x == 5 else 0)

# 전처리된 DataFrame 저장
df.to_csv(output_file, index=False)

print(f"전처리 완료 및 파일 저장됨 -> {output_file}")