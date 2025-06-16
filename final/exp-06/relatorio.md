```python
def build_unet_colorization(input_shape=(128, 128, 1)):
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2,2))(c1)

    c2 = Conv2D(128, (3,3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2,2))(c2)

    c3 = Conv2D(256, (3,3), activation='relu', padding='same')(p2)
    p3 = MaxPooling2D((2,2))(c3)

    # Bottleneck
    bn = Conv2D(512, (3,3), activation='relu', padding='same')(p3)

    # Decoder
    u1 = UpSampling2D((2,2))(bn)
    u1 = concatenate([u1, c3])
    c4 = Conv2D(256, (3,3), activation='relu', padding='same')(u1)

    u2 = UpSampling2D((2,2))(c4)
    u2 = concatenate([u2, c2])
    c5 = Conv2D(128, (3,3), activation='relu', padding='same')(u2)

    u3 = UpSampling2D((2,2))(c5)
    u3 = concatenate([u3, c1])
    c6 = Conv2D(64, (3,3), activation='relu', padding='same')(u3)

    outputs = Conv2D(2, (3,3), activation='tanh', padding='same')(c6)

    return Model(inputs, outputs)
```