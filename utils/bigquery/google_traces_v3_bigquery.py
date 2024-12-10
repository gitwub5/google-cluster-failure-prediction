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

# 기준 시작 시간과 30분 동안의 데이터를 설정 (예시: 특정 시간)
start_time = 300000000  # 예시: 특정 start_time 값 (마이크로초 단위)
end_time = start_time + (1 * 60 * 60 * 1_000_000)  # 6  * 1 시간 후의 end_time 값

# 1. 주 쿼리 작성
query = f"""
    SELECT 
        iu.machine_id,
        iu.start_time,
        iu.end_time,
        iu.average_usage.cpus AS average_usage_cpus,
        iu.average_usage.memory AS average_usage_memory,
        iu.maximum_usage.cpus AS maximum_usage_cpus,
        iu.maximum_usage.memory AS maximum_usage_memory,
        iu.assigned_memory,
        iu.page_cache_memory,
        ce.collection_id, 
        ce.time AS event_time,
        ce.type AS event_type
    FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_usage` AS iu
    LEFT JOIN `google.com:google-cluster-data.clusterdata_2019_a.collection_events` AS ce
    ON iu.collection_id = ce.collection_id
    WHERE iu.start_time >= {start_time} 
      AND iu.end_time <= {end_time}
      AND ce.type IN (3, 4, 5, 6, 7, 8)
    ORDER BY iu.start_time
"""

# 2. 검증 쿼리 작성: 하나의 `Instance Usage`에 여러 `Collection Event`가 매핑되는지 확인
validation_query = """
    SELECT iu.collection_id, COUNT(*) AS event_count
    FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_usage` AS iu
    JOIN `google.com:google-cluster-data.clusterdata_2019_a.collection_events` AS ce
    ON iu.collection_id = ce.collection_id
    GROUP BY iu.collection_id
    HAVING COUNT(*) > 1
"""
#
# 3. 주 쿼리 실행
print("Executing main query...")
query_job = client.query(query)
results = query_job.result()
print("Main query executed successfully.")

# # 4. 검증 쿼리 실행
# print("Executing validation query...")
# validation_job = client.query(validation_query)
# validation_results = validation_job.result()
# print("Validation query executed successfully.")

# # 5. 검증 결과 확인 및 출력
# multiple_events_found = False
# for row in validation_results:
#     multiple_events_found = True
#     print(f"Collection ID: {row.collection_id}, Event Count: {row.event_count}")
#
# if not multiple_events_found:
#     print("All Instance Usage records have exactly one or zero Collection Event.")

# 6. 주 쿼리 결과를 CSV 파일로 저장
output_dir = '../../data/google_traces_v3'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "output_data2.csv")

with open(output_file, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)

    # 헤더 작성
    headers = [
        "machine_id", "start_time", "end_time", "average_usage_cpus", "average_usage_memory",
        "maximum_usage_cpus", "maximum_usage_memory", "assigned_memory", "page_cache_memory",
        "collection_id", "event_time", "event_type"
    ]
    writer.writerow(headers)

    # 데이터 작성
    for row in results:
        writer.writerow([
            row.machine_id, row.start_time, row.end_time, row.average_usage_cpus,
            row.average_usage_memory, row.maximum_usage_cpus, row.maximum_usage_memory,
            row.assigned_memory, row.page_cache_memory, row.collection_id,
            row.event_time, row.event_type
        ])

print(f"Main query results saved to {output_file}")