import torch
print(torch.__version__)
print(torch.cuda.is_available())

# создаем квадратный тензор размером 2 х 3 ячейки
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.uint8)
print(tensor_2d.shape)

# добавим еще одну ось, сделав тензор 3-х мерным
tensor_3d = tensor_2d.unsqueeze(0)
print(tensor_3d.shape)

# transpose
print(tensor_3d)
tensor_3d = tensor_3d.transpose(2, 1)
print(tensor_3d)
print(tensor_3d.shape)

# transpose
tensor_3d = tensor_3d.reshape(1, 2, 3)
print(tensor_3d)
print(tensor_3d.shape)

# объединим копии тензоров по 0-й оси
batch_tensor_3d = torch.concat([tensor_3d]*16, dim=0)
print(batch_tensor_3d.shape)
print(batch_tensor_3d)

# перемещение тензора между ГПУ и ЦПУ
print(batch_tensor_3d.device)
batch_tensor_3d = batch_tensor_3d.to('cuda')
print(batch_tensor_3d.device)
batch_tensor_3d = batch_tensor_3d.cpu()
print(batch_tensor_3d.device)

# Создаем тензор с requires_grad=True
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Выполняем операции с тензором
y = x ** 2
z = y.sum()

# Выполняем обратное распространение (backward pass)
z.backward()

# Выводим градиенты
print("Значение тензора x:", x)
print("Значение тензора y:", y)
print("Значение тензора z:", z)
print("Градиенты тензора x:", x.grad)