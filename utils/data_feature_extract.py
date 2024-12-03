import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks

# CSV 파일 경로 설정
input_file = '../data/google_traces_v3/output_data.csv'
output_file = '../data/google_traces_v3/preprocessed_dta.csv'

# CSV 파일 읽기
df = pd.read_csv(input_file)

# <추가 Feature 생성 - 각 machine_id별로 적용>
def extract_features(group):
    # 이동 평균
    group['ma_average_usage_cpus'] = group['average_usage_cpus'].rolling(window=5, min_periods=1).mean()
    group['ma_average_usage_memory'] = group['average_usage_memory'].rolling(window=5, min_periods=1).mean()

    # 차분
    group['diff_maximum_usage_cpus'] = group['maximum_usage_cpus'].diff().fillna(0)

    return group

# machine_id별로 그룹화하여 특징 생성
df = df.groupby('machine_id', group_keys=False).apply(extract_features)

print(f"특징 추가 및 추출 성공")

# 전처리된 DataFrame 저장
df.to_csv(output_file, index=False)

print(f"전처리 완료 및 파일 저장됨 -> {output_file}")