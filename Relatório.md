# Relatório CIFAR-10

## Autores: Gabriel de Sousa e Suellen Oliveira

### Modelo Inicial: lenet_5

```python
def CNN_model(input_shape=(32, 32, 3), num_classes=10):
  model = Sequential()
  model.add(Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=input_shape))
  model.add(AvgPool2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Conv2D(16, kernel_size=(5, 5), activation='tanh'))
  model.add(AvgPool2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Conv2D(120, kernel_size=(5, 5), activation='tanh'))

  model.add(Flatten())
  model.add(Dense(84, activation='tanh'))
  model.add(Dense(num_classes, activation='softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model
```

## Curva de Aprendizado
![alt text](image.png)

## Matriz de Confusão
![alt text](image-1.png)

## F1-score, Recall e Precision
> Recall:  0.5227

> Precision:  0.5167354741039393

> F1 Score:  0.5197006241238221

## Sensibilidade(TPR) e Especifidade(TNR)

## Acurácia ponderada e Curva ROC (com a AUC)
![alt text](image-2.png)

### Modelo Melhorado: (lenet_50)