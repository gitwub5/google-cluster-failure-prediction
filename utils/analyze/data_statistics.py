import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 읽기 (csv_file 경로 수정해줘야함)
csv_file = '../../data/google_traces_v3/instance_usage_preprocessed.csv'
df = pd.read_csv(csv_file)

# pandas 출력 옵션 설정 (전체 내용 출력)
pd.set_option('display.max_columns', None)  # 모든 열 출력
pd.set_option('display.max_rows', None)     # 모든 행 출력
pd.set_option('display.float_format', '{:.6f}'.format)  # 소수점 형식 설정

# 선택된 열
columns_to_analyze = [
    'average_usage_cpus', 'average_usage_memory', 'maximum_usage_cpus',
    'maximum_usage_memory'
]

# 기술 통계 계산
statistics = df[columns_to_analyze].describe()

# 기술 통계 출력
print(statistics)

# 기술 통계를 CSV 파일로 저장
output_file = '../../data/google_traces_v3/analyze/techical_statistics_summary2.csv'
statistics.to_csv(output_file, index=True)
print(f"Technical statistics saved to {output_file}")

# 고유한 machine_id, collection_id, event_type의 개수 찾기
unique_machine_ids = df['machine_id'].nunique()

print(f"Unique machine IDs: {unique_machine_ids}")

# 상관계수 계산
correlation_matrix = df[columns_to_analyze].corr()

# 상관계수 출력
print("Correlation Matrix:")
print(correlation_matrix)

# 상관계수를 CSV 파일로 저장
correlation_output_file = '../../data/google_traces_v3/analyze/correlation_matrix2.csv'
correlation_matrix.to_csv(correlation_output_file, index=True)
print(f"Correlation matrix saved to {correlation_output_file}")

# 상관계수 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()

# 히트맵 저장
heatmap_output_file = '../../data/google_traces_v3/analyze/correlation_heatmap2.png'
plt.savefig(heatmap_output_file)
print(f"Correlation heatmap saved to {heatmap_output_file}")

# 히트맵 표시
plt.show()