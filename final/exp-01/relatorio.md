```python
def build_autoencoder(input_shape=(150, 150, 3)):
    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.1)(x)

    encoded = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    x = Cropping2D(((1, 1), (1, 1)))(x)

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    return Model(input_img, decoded)


autoencoder = build_autoencoder()
autoencoder.compile(optimizer=Adam(1e-3), loss='mse')
autoencoder.summary()
```

```python
history = autoencoder.fit(
    x=X_train, y=Y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, Y_test),
    callbacks=[early_stop]
)
```