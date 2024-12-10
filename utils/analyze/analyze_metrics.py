import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 로드
metrics_save_path = '../../data/metrics/metrics.csv'
metrics_df = pd.read_csv(metrics_save_path)

# 데이터 확인
print("데이터 샘플:")
print(metrics_df.head())

# 그래프 시각화 함수
def plot_metrics(metrics_df):
    # Epoch vs Loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['epoch'], metrics_df['loss'], label='Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.show()

    # Epoch vs MSE
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['epoch'], metrics_df['mse'], label='MSE', color='orange')
    plt.title('Mean Squared Error (MSE) over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid()
    plt.legend()
    plt.show()

    # Epoch vs RMSE
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['epoch'], metrics_df['rmse'], label='RMSE', color='green')
    plt.title('Root Mean Squared Error (RMSE) over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid()
    plt.legend()
    plt.show()

    # Epoch vs R²
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['epoch'], metrics_df['r2'], label='R²', color='red')
    plt.title('R² Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.grid()
    plt.legend()
    plt.show()

# 시각화 실행
plot_metrics(metrics_df)