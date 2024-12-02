import os
from google.cloud import bigquery
import csv
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수에서 GCP 프로젝트 이름 가져오기
project_name = os.getenv("GCP_PROJECT")

# BigQuery 클라이언트 생성
client = bigquery.Client(project=project_name)

# 디렉터리 설정
destination_dir = '../data/google_traces_v3'
os.makedirs(destination_dir, exist_ok=True)  # 디렉터리가 없으면 생성

# 파일 경로 설정
output_file = os.path.join(destination_dir, "output_data.csv")

# 기준 시작 시간과 30분 동안의 데이터를 설정 (예시: 특정 시간)
start_time = 300000000  # 예시: 특정 start_time 값 (마이크로초 단위)
end_time = start_time + (6 * 60 * 60 * 1_000_000)  # 6  * 1 시간 후의 end_time 값

# 쿼리 작성
query = f"""
    SELECT 
        ce.collection_id, 
        ce.time AS event_time,
        ce.type AS event_type, 
        iu.machine_id,
        iu.average_usage.cpus AS average_usage_cpus,
        iu.average_usage.memory AS average_usage_memory,
        iu.maximum_usage.cpus AS maximum_usage_cpus,
        iu.maximum_usage.memory AS maximum_usage_memory,
        iu.assigned_memory,
        iu.page_cache_memory
    FROM `google.com:google-cluster-data.clusterdata_2019_a.collection_events` AS ce
    JOIN `google.com:google-cluster-data.clusterdata_2019_a.instance_usage` AS iu
    ON ce.collection_id = iu.collection_id
    WHERE ce.time >= {start_time} AND ce.time <= {end_time} AND ce.type IN (3, 4, 5, 6, 7, 8)
    ORDER BY ce.time
"""

# 쿼리 실행
query_job = client.query(query)
results = query_job.result()

# CSV 파일로 저장
with open(output_file, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)

    # 헤더 작성
    headers = [
        "collection_id", "event_time", "event_type", "machine_id",
        "average_usage_cpus", "average_usage_memory", "maximum_usage_cpus", "maximum_usage_memory",
        "assigned_memory", "page_cache_memory"
    ]
    writer.writerow(headers)

    # 데이터 평면화하여 작성
    for row in results:
        row_data = [
            row.collection_id, row.event_time, row.event_type, row.machine_id,
            row.average_usage_cpus, row.average_usage_memory,
            row.maximum_usage_cpus, row.maximum_usage_memory,
            row.assigned_memory, row.page_cache_memory
        ]
        writer.writerow(row_data)

print(f"Data saved to {output_file}")