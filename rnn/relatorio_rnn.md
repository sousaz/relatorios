
# Relatório – RNN para Previsão de Temperaturas

**Autor:** Gabriel Sousa  
**Dataset:** *Daily Minimum Temperatures in Melbourne*

---

## 1. Introdução

O objetivo deste trabalho é construir um modelo baseado em Redes Neurais Recorrentes (RNN), para prever temperaturas mínimas diárias. A previsão de séries temporais é uma tarefa importante em diversos domínios, e este estudo visa explorar a capacidade das redes neurais em capturar padrões temporais.

---

## 2. Descrição da Base de Dados

A base de dados utilizada contém registros diários de temperaturas mínimas na cidade de Melbourne, Austrália.  
Cada entrada representa a menor temperatura registrada em um determinado dia.

![Exemplo dos dados do dataset]([image.png](https://github.com/sousaz/relatorios/blob/main/rnn/image.png?raw=true))
---

## 3. Pré-processamento dos Dados

### 3.1 Normalização

A normalização foi aplicada para converter os dados para uma escala comum (entre 0 e 1), utilizando a técnica Min-Max:

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df['MinTemp'] = scaler.fit_transform(df[['MinTemp']])
```

### 3.2 Transformação da Série Temporal em Formato Supervisionado

A série temporal foi convertida em formato supervisionado com lag de 14 dias, permitindo que o modelo aprenda com duas semanas de histórico:

```python
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

supervised = timeseries_to_supervised(df['MinTemp'], 14)
```

### 3.3 Separação de Dados

Dividiu-se os dados em 80% para treino e 20% para teste. Os dados foram reshaped para o formato esperado pela rede LSTM.

---

## 4. Planejamento dos Experimentos

### Objetivo Geral

Avaliar a capacidade de uma rede LSTM para prever temperaturas mínimas diárias com base em dados históricos.

### Variáveis Consideradas nos Experimentos

- Quantidade de dias de histórico (lag).
- Estrutura da rede neural (número de camadas e neurônios).
- Regularização com Dropout.
- Estratégias de parada precoce (early stopping).

---

## 5. Experimentos Realizados

### Experimento 1: Rede LSTM com lag = 14

**Arquitetura:**
```python
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

**Descrição:**

Foi treinada uma rede LSTM com 3 camadas, utilizando um histórico de 14 dias para prever o próximo valor.

**Resultados:**

| Métrica   | Valor   |
|-----------|---------|
| MAE       | 1.9159  |
| MSE       | 6.0083  |
| RMSE      | 2.4511  |
| R² Score  | 0.6432  |

**Analise e Discussão:**

O modelo apresentou um MAE de aproximadamente 1,92 graus e RMSE de 2,45. O R² de 0,64 indica que o modelo explica cerca de 64% da variação nos dados. A arquitetura simples foi capaz de capturar padrões básicos, mas o desempenho poderia ser melhorado.

### Experimento 2: Rede LSTM e Dropout com Lag = 14

**Arquitetura:**
```python
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```
**Descrição:**

Foi treinada uma rede LSTM com 3 camadas, utilizando um histórico de 14 dias para prever o próximo valor. Adicionou-se Dropout entre as camadas para previnir overfitting.

**Resultados:**

| Métrica   | Valor   |
|-----------|---------|
| MAE       | 1.9363  |
| MSE       | 6.1615  |
| RMSE      | 2.4822  |
| R² Score  | 0.6341  |

**Análise e Discussão:**

O modelo com Dropout apresentou um MAE de aproximadamente 1,94 graus e RMSE de 2,48. O R² de 0,63 indica que o modelo explica cerca de 63% da variação nos dados. A adição de Dropout não trouxe melhorias significativas em relação ao modelo anterior.

### Experimento 3: Rede LSTM Bidirecional com Lag = 14

**Arquitetura:**
```python
model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

**Descrição:**

Foi treinada uma rede LSTM bidirecional com 3 camadas, utilizando um histórico de 14 dias para prever o próximo valor. O uso de Bidirectional permite que a rede aprenda tanto a partir do passado quanto do futuro.

**Resultados:**

| Métrica   | Valor   |
|-----------|---------|
| MAE       | 1.7507  |
| MSE       | 4.9293  |
| RMSE      | 2.2202  |
| R² Score  | 0.7073  |

**Análise e Discussão:**  
O modelo bidirecional apresentou um MAE de aproximadamente 1,75 graus e RMSE de 2,22. O R² de 0,71 indica que o modelo explica cerca de 71% da variação nos dados. A arquitetura bidirecional melhorou o desempenho em comparação com os modelos anteriores.

O uso de camadas bidirecionais ajudou a capturar dependências tanto passadas quanto futuras na sequência, e o uso de Dropout contribuiu para reduzir o overfitting.

---

## 6. Discussão dos experimentos

Os experimentos realizados mostraram que a adição de Dropout não trouxe melhorias significativas no desempenho do modelo. A arquitetura bidirecional, por outro lado, demonstrou ser mais eficaz na captura de padrões temporais.

Além disso, foi testado os modelos com lags diferentes como 30(1 mês), mas não houve melhorias significativas. O modelo com lag de 14 dias foi o mais eficiente.

E também foi allterado os numeros de neurônios e camadas, mas o modelo com 3 camadas e 128 neurônios na primeira camada foi o mais eficiente(experimento 3).

---

## 7. Resultados Obtidos

Os resultados obtidos indicam que a arquitetura do experimento 3 foi a mais eficiente para a tarefa. Com um MAE de 1,75 e um R² de 0,71, o modelo foi capaz de prever temperaturas mínimas com uma precisão razoável.

![Real vs Previsto]([image-1.png](https://github.com/sousaz/relatorios/blob/main/rnn/image-1.png?raw=true))

![Curva de aprendizado]([image-2.png](https://github.com/sousaz/relatorios/blob/main/rnn/image-2.png?raw=true))

---

## 8. Considerações Finais

O trabalho demonstrou a eficácia das redes LSTM na previsão de séries temporais, especialmente em tarefas relacionadas a dados meteorológicos. A arquitetura bidirecional se destacou, mostrando que a consideração de informações passadas e futuras pode melhorar significativamente o desempenho do modelo.

A normalização dos dados e a transformação da série temporal em formato supervisionado foram etapas cruciais para o sucesso do modelo.

A pesquisa futura pode explorar outras arquiteturas, como GRU, e técnicas de ensemble para melhorar ainda mais a precisão das previsões.

---
