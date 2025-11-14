import pandas as pd
import numpy as np

# 주어진 데이터
data = {
    'landmark': ['left arm', 'left shoulder', 'right arm', 'right shoulder', 'right knee', 'right hip', 'left knee', 'left hip'],
    'avg_error_cm': [2.35, 2.12, 1.01, 2.00, 1.44, 7.29, 5.72, 5.40]
}

# 데이터프레임 생성
df = pd.DataFrame(data)

# 결과 출력
print(df)

# 평균 출력
average_error = np.mean(df['avg_error_cm'])
print(f"\nAverage: {average_error:.2f} cm")