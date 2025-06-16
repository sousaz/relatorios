```python
def build_autoencoder2(input_shape=(128, 128, 3)):
    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # (150, 150, 32)
    x = MaxPooling2D((2, 2), padding='same')(x)  # (75, 75, 32)

    encoded = Conv2D(256, (3, 3), activation='relu', padding='same')(x)  # (75, 75, 256)

    # Decoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)  # (75, 75, 32)
    x = UpSampling2D((2, 2))(x)  # (150, 150, 32)

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # (150, 150, 3)

    return Model(input_img, decoded)
```