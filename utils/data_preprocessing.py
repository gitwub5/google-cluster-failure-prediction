import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks

# CSV 파일 경로 설정
input_file = '../data/google_traces_v3/csv/output_data.csv'
output_file = '../data/google_traces_v3/output_data3_preprocessed.csv'

# CSV 파일 읽기
df = pd.read_csv(input_file)

# <결측값 처리 & 이상치 처리>
# NaN  or NULL, Inf 값 제거
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# 특정 열 중 하나라도 0인 행 제거
columns_to_check = [
    'average_usage_cpus', 'average_usage_memory', 'maximum_usage_cpus',
    'maximum_usage_memory', 'random_sample_usage_cpus', 'assigned_memory',
    'page_cache_memory'
]
df = df[(df[columns_to_check] != 0).all(axis=1)]

print(f"결측값 처리 & 이상치 처리 성공")


# <Min-Max 정규화>
columns_to_normalize = [
    'average_usage_cpus', 'average_usage_memory', 'maximum_usage_cpus',
    'maximum_usage_memory', 'random_sample_usage_cpus', 'assigned_memory',
    'page_cache_memory'
]

scaler = MinMaxScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

print(f"Min-Max 정규화 성공")

# <추가 Feature 생성 - 각 machine_id별로 적용>
def extract_features(group):
    # 이동 평균
    group['ma_average_usage_cpus'] = group['average_usage_cpus'].rolling(window=5, min_periods=1).mean()
    group['ma_average_usage_memory'] = group['average_usage_memory'].rolling(window=5, min_periods=1).mean()

    # 차분
    group['diff_maximum_usage_cpus'] = group['maximum_usage_cpus'].diff().fillna(0)

    # # 피크 탐지
    # peaks, _ = find_peaks(group['assigned_memory'], height=0)
    # if len(peaks) > 0:
    #     # `peaks`를 그룹의 원래 인덱스로 변환
    #     original_indices = group.index[peaks]
    #     print(f"Debug: Original indices of peaks: {original_indices}")
    #
    #     # 피크가 있는 위치에 1 설정
    #     group.loc[original_indices, 'peak_assigned_memory'] = 1
    # else:
    #     print("Debug: No peaks detected for this group.")

    return group

# machine_id별로 그룹화하여 특징 생성
df = df.groupby('machine_id', group_keys=False).apply(extract_features)

# Failed 열 추가 (event_type이 5인 경우 1, 아니면 0)
df['Failed'] = df['event_type'].apply(lambda x: 1 if x == 5 else 0)

print(f"특징 추가 및 추출 성공")

# 전처리된 DataFrame 저장
df.to_csv(output_file, index=False)

print(f"전처리 완료 및 파일 저장됨 -> {output_file}")