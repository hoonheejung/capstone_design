import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import memtorch

# 데이터 로드
data = torch.load("multi_wafer_data_64.pth", weights_only=False)
X_train, y_train, classes = data['X_train'], data['y_train'], data['classes']

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(180) 
])

class WaferDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X, self.y, self.transform = X, y, transform
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.transform: x = self.transform(x)
        return x, y

# 64x64 전용 모델 구조
class RobustMultiWaferCNN_64(nn.Module):
    def __init__(self, num_classes):
        super(RobustMultiWaferCNN_64, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU()
        )
        # 64 -> 32 -> 16 (stride 2씩 두 번)
        self.fc = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = RobustMultiWaferCNN_64(len(classes))
TaO2_params = {'r_on_variation': 0.05, 'r_off_variation': 0.10}

memristive_model = memtorch.patch_model(model, 
                                        memristor_model=memtorch.bh.memristor.VTEAM,
                                        memristor_model_params=TaO2_params,
                                        tile_shape=(128, 128), adc_bitwidth=8, dac_bitwidth=8)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(memristive_model.parameters(), lr=0.001)
train_loader = DataLoader(WaferDataset(X_train, y_train, transform=train_transform), batch_size=32, shuffle=True)

print("🏋️ 64x64 Robust 학습 시작...")
for epoch in range(12):
    memristive_model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(memristive_model(inputs), labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/12 완료")

torch.save(memristive_model.state_dict(), "multi_memristor_model_64.pth")