from google.cloud import bigquery
import csv
import os

# GCP 프로젝트 설정 (환경 변수에서 프로젝트 이름 가져오기)
project_name = os.getenv("GCP_PROJECT")
client = bigquery.Client(project=project_name)

# 파라미터 설정
num_machine_ids = 200  # 랜덤으로 가져올 machine_id 개수
start_time = 300000000  # 시작 시간 (마이크로초 단위)
end_time = start_time + (7 * 24 * 60 * 60 * 1_000_000) # 일주일 간의 데이터

# 랜덤한 machine_id 선택 쿼리
random_machine_query = f"""
    SELECT DISTINCT machine_id
    FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_usage`
    WHERE start_time >= {start_time} AND end_time <= {end_time}
    ORDER BY RAND()
    LIMIT {num_machine_ids}
"""

# 랜덤 machine_id 쿼리 실행
print("Fetching random machine IDs...")
machine_id_job = client.query(random_machine_query)
machine_id_results = machine_id_job.result()
random_machine_ids = [row.machine_id for row in machine_id_results]

if not random_machine_ids:
    raise ValueError("No machine IDs found in the specified time range.")

print(f"Randomly selected machine IDs: {random_machine_ids}")

# 데이터 필터링 쿼리
query = f"""
    SELECT 
        machine_id,
        start_time,
        end_time,
        collection_id,
        alloc_collection_id,
        average_usage.cpus AS average_usage_cpus,
        average_usage.memory AS average_usage_memory,
        maximum_usage.cpus AS maximum_usage_cpus,
        maximum_usage.memory AS maximum_usage_memory,
        assigned_memory,
        page_cache_memory
    FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_usage`
    WHERE machine_id IN UNNEST({random_machine_ids})
      AND start_time >= {start_time}
      AND end_time <= {end_time}
    ORDER BY machine_id, start_time
"""

# 데이터 쿼리 실행
print("Fetching data for selected machine IDs...")
query_job = client.query(query)
results = query_job.result()

# 결과를 CSV로 저장
output_dir = '../../data/google_traces_v3'
os.makedirs(output_dir, exist_ok=True)  # 디렉터리가 없으면 생성
output_file = os.path.join(output_dir, 'instance_usage_data.csv')

with open(output_file, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)

    # 헤더 작성
    headers = [
        "machine_id", "start_time", "end_time",
        "collection_id", "alloc_collection_id",
        "average_usage_cpus", "average_usage_memory",
        "maximum_usage_cpus", "maximum_usage_memory",
        "assigned_memory", "page_cache_memory",
    ]
    writer.writerow(headers)

    # 데이터 작성
    for row in results:
        writer.writerow([
            row.machine_id, row.start_time, row.end_time,
            row.collection_id, row.alloc_collection_id,
            row.average_usage_cpus, row.average_usage_memory,
            row.maximum_usage_cpus, row.maximum_usage_memory,
            row.assigned_memory, row.page_cache_memory,
        ])

print(f"Main query results saved to {output_file}")