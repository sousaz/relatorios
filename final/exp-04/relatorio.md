
```python
def build_autoencoder(input_shape=(128, 128, 3)):
    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # 150x150
    x = MaxPooling2D((2, 2), padding='same')(x)  # 75x75
    x = Dropout(0.1)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)  # 75x75
    x = MaxPooling2D((2, 2), padding='same')(x)  # 38x38
    x = Dropout(0.1)(x)

    encoded = Conv2D(128, (3, 3), activation='relu', padding='same')(x)  # 38x38

    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)  # 38x38
    x = UpSampling2D((2, 2))(x)  # 76x76

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # 76x76
    x = UpSampling2D((2, 2))(x)  # 152x152

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # 150x150x3

    return Model(input_img, decoded)
```