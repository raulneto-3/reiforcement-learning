import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Verificar se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset fictício (substitua pelos seus dados reais)
np.random.seed(0)
X = np.random.rand(1000, 3)  # 1000 amostras, 3 características (ex: tamanho, número de quartos, idade)
y = X @ np.array([300000, 50000, -10000]) + 50000  # Preço fictício

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Converter para tensores do PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Definindo o modelo de regressão
class HousePriceModel(nn.Module):
    def __init__(self):
        super(HousePriceModel, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Função para treinar o modelo
def train(model, X_train, y_train, criterion, optimizer, epochs=100):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train).flatten()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Função para avaliar o modelo
def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).flatten()
        mse = mean_squared_error(y_test.cpu(), predictions.cpu())
        mae = mean_absolute_error(y_test.cpu(), predictions.cpu())
        print(f'Mean Squared Error: {mse:.4f}')
        print(f'Mean Absolute Error: {mae:.4f}')
        return predictions

# Inicializar o modelo, critério de perda e otimizador
model = HousePriceModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Treinar o modelo
train(model, X_train, y_train, criterion, optimizer, epochs=100)

# Avaliar o modelo
predictions = evaluate(model, X_test, y_test)

# Reverter a normalização para exibir os resultados reais
y_test_real = scaler_y.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).flatten()
predictions_real = scaler_y.inverse_transform(predictions.cpu().numpy().reshape(-1, 1)).flatten()

# Exibir alguns dos resultados mais próximos e mais distantes
differences = np.abs(y_test_real - predictions_real)
sorted_indices = np.argsort(differences)

print("\nResultados mais próximos:")
for i in sorted_indices[:5]:
    print(f"Real: {y_test_real[i]:.2f}, Predito: {predictions_real[i]:.2f}, Diferença: {differences[i]:.2f}")

print("\nResultados mais distantes:")
for i in sorted_indices[-5:]:
    print(f"Real: {y_test_real[i]:.2f}, Predito: {predictions_real[i]:.2f}, Diferença: {differences[i]:.2f}")

# Plotar os resultados
plt.figure(figsize=(10, 5))
plt.scatter(y_test_real, predictions_real, alpha=0.5)
plt.plot([min(y_test_real), max(y_test_real)], [min(y_test_real), max(y_test_real)], color='red')
plt.xlabel('Preços Reais')
plt.ylabel('Preços Preditos')
plt.title('Preços Reais vs Preços Preditos')
plt.show()