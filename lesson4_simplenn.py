import torch
import torch.nn as nn
import torch.onnx

# 1. Определим базовую нейронную сеть
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 2 класса
        )

    def forward(self, x):
        return self.net(x)

def train(lr=0.01, epochs = 100, batch_size = 64):
    # 2. Создадим модель, входные данные, и обучим её немного
    model = SimpleClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Print the number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")

    # Генерируем случайные входы и метки
    x = torch.randn(batch_size, 10)  # 64 объекта, 10 признаков
    y = torch.randint(0, 2, (batch_size,))  # метки: 0 или 1

    # Простое обучение (N эпохи)
    for epoch in range(epochs):
        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # 3. Экспорт в ONNX
    dummy_input = torch.randn(1, 10)
    torch.onnx.export(
        model,
        dummy_input,
        "simple_classifier.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True
    )

    print("Модель экспортирована в simple_classifier.onnx")

if __name__ == "__main__":
    train()