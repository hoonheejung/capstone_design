import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import memtorch
import numpy as np
import os

# 1. 검증용 데이터 로드
print("📦 64x64 검증용 데이터 로딩 중...")
try:
    # Step 1에서 생성한 64x64 전용 데이터 파일을 불러옵니다.
    data = torch.load("multi_wafer_data_64.pth", weights_only=False)
    X_test = data['X_test']
    y_test = data['y_test']
    classes = data['classes']
    print(f"✅ 데이터 로드 성공! (테스트 샘플: {len(X_test)}개)")
except FileNotFoundError:
    print("❌ 'multi_wafer_data_64.pth' 파일을 찾을 수 없습니다. Step 1을 먼저 실행하세요.")
    exit()

# 2. 64x64 전용 모델 구조 정의 (Step 2와 100% 동일해야 가중치 로드 가능)
class RobustMultiWaferCNN_64(nn.Module):
    def __init__(self, num_classes):
        super(RobustMultiWaferCNN_64, self).__init__()
        self.features = nn.Sequential(
            # Layer 1: 64x64 -> 32x32
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            
            # Layer 2: 32x32 -> 16x16
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # 64 해상도에서 최종 특징 맵 크기는 16x16입니다.
        self.fc = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 3. 멤리스터 모델 구성 및 학습된 가중치 로드
print("⚙️ TaO2 기반 멤리스터 가속기 구성 및 가중치 로드 중...")
model = RobustMultiWaferCNN_64(len(classes))

# 학습 때와 동일한 물리적 변동성(Variation) 파라미터를 주입합니다.
TaO2_params = {
    'r_on_variation': 0.05,
    'r_off_variation': 0.10,
}

memristive_model = memtorch.patch_model(model, 
                                        memristor_model=memtorch.bh.memristor.VTEAM,
                                        memristor_model_params=TaO2_params,
                                        tile_shape=(128, 128), 
                                        adc_bitwidth=8, 
                                        dac_bitwidth=8)

try:
    # Step 2에서 학습 완료된 64x64 모델 파일을 불러옵니다.
    memristive_model.load_state_dict(torch.load("multi_memristor_model_64.pth", weights_only=False))
    memristive_model.eval() # 평가 모드 전환
    print("✅ 학습된 모델 로드 성공!")
except FileNotFoundError:
    print("❌ 'multi_memristor_model_64.pth'가 없습니다. Step 2 학습을 먼저 완료하세요.")
    exit()

# 4. 성능 테스트 수행
all_preds = []
all_labels = []
threshold = 0.5 # 멀티 라벨 판정 기준값

print("🧐 64x64 기반 혼합 결함 정밀 진단 시작...")
with torch.no_grad():
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
    for inputs, labels in test_loader:
        outputs = memristive_model(inputs)
        # Sigmoid를 통과시켜 각 클래스별 확률값이 0.5를 넘으면 결함으로 판정
        predicted = (torch.sigmoid(outputs) > threshold).int()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# 5. 최종 성능 분석 보고서 출력
print("\n" + "="*60)
print("📊 [TaO2 기반 PIM 가속기: 64x64 고해상도 결함 분석 보고서]")
print("="*60)

# 결함 유형별 상세 지표 (Accuracy, Precision, Recall)
mcm = multilabel_confusion_matrix(all_labels, all_preds)

print(f"{'Defect Type':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10}")
print("-" * 65)

for i, class_name in enumerate(classes):
    tn, fp, fn, tp = mcm[i].ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"{class_name:<15} | {accuracy:>8.1f}% | {precision:>9.2f} | {recall:>9.2f}")

print("-" * 65)
print("\n📝 [전체 성능 요약 (Classification Report)]")
print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))

# 6. 시각화: 주요 결함 유형별 Confusion Matrix
# Edge-Loc의 개선 여부와 Scratch의 정밀도를 확인합니다.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

targets_to_plot = ['Edge-Loc', 'Scratch']
for i, target in enumerate(targets_to_plot):
    idx = list(classes).index(target)
    sns.heatmap(mcm[idx], annot=True, fmt='d', cmap='Greens', ax=axes[i],
                xticklabels=['Pred: No', 'Pred: Yes'], 
                yticklabels=['Actual: No', 'Actual: Yes'])
    axes[i].set_title(f'{target} Detection Result (64x64)')

plt.tight_layout()
plt.show()

print("\n✅ 모든 평가 및 시각화가 완료되었습니다.")