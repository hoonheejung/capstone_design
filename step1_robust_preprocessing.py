import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# 1. 데이터 로드 (52x52 해상도 원본)
raw_data = np.load("../archive (1)/Wafer_Map_Datasets.npz")
X_raw = raw_data['arr_0'] 
y_raw = raw_data['arr_1'] 

# 2. 64x64로 리사이징 (Bilinear interpolation 사용)
X_temp = torch.FloatTensor(X_raw.astype(np.float32) / 2.0).unsqueeze(1) # [N, 1, 52, 52]
X_64 = F.interpolate(X_temp, size=(64, 64), mode='bilinear', align_corners=False)
y_tensor = torch.FloatTensor(y_raw)

print(f"📏 리사이징 완료: {X_temp.shape[2:]} -> {X_64.shape[2:]}")

# 3. 데이터 밸런싱 (정상 데이터 조절)
is_normal = np.all(y_raw == 0, axis=1)
X_normal = X_64[is_normal]
y_normal = y_tensor[is_normal]
X_defect = X_64[~is_normal]
y_defect = y_tensor[~is_normal]

sample_size = min(len(X_normal), int(len(X_defect) * 1.5))
indices = np.random.choice(len(X_normal), sample_size, replace=False)

X_balanced = torch.cat([X_normal[indices], X_defect], dim=0)
y_balanced = torch.cat([y_normal[indices], y_defect], dim=0)

# 4. 학습/검증 분리 및 저장
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
all_classes = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-Full', 'Scratch', 'Random']

torch.save({
    'X_train': X_train, 'X_test': X_test,
    'y_train': y_train, 'y_test': y_test,
    'classes': np.array(all_classes)
}, "multi_wafer_data_64.pth")
print("✅ [Step 1] 64x64 전처리 및 저장 완료!")