'''
Author: hibana2077 hibana2077@gmail.com
Date: 2023-07-30 15:28:04
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2023-07-30 16:39:20
FilePath: \ST_LAB\lab\vit_ex\vit.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import MultiHeadAttention

def create_vit_architecture():
    num_classes = 10
    num_patches = 16
    projection_dim = 64
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 8
    mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

    inputs = layers.Input(shape=(28, 28, 1))
    # Create patches.
    images_patches = layers.Conv2D(
        filters=num_patches, kernel_size=7, strides=7, padding="valid"
    )(inputs)
    # Encode patches.
    encoded_patches = layers.Dense(units=projection_dim)(images_patches)
    # Reshape patches to be fed into transformer block as a sequence.
    patches_reshaped = layers.Reshape((num_patches, projection_dim))(encoded_patches)

    # Create positional embeddings.
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim
    )(positions)
    # print(position_embedding.shape)
    # print(f"reshape to {num_patches, projection_dim}")
    # position_embedding = layers.Reshape((num_patches, projection_dim))(
    #     position_embedding
    # )

    # Add positional embeddings to the patch encoding.
    patches_with_position = layers.Add()([patches_reshaped, position_embedding])

    # Transformer layers
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(patches_with_position)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, patches_with_position])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3_reshaped = layers.Reshape((-1, x3.shape[-1]))(x3)
        mlp_output = layers.Dense(units=transformer_units[0], activation=tf.nn.gelu)(
            x3_reshaped
        )
        mlp_output = layers.Dense(units=transformer_units[1])(mlp_output)
        mlp_output_reshaped = layers.Reshape((-1, num_patches, transformer_units[1]))(
            mlp_output
        )
        # Skip connection 2.
        patches_with_position = layers.Add()([mlp_output_reshaped, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(patches_with_position)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    # Classifier head
    logits = layers.Dense(mlp_head_units[0], activation=tf.nn.gelu)(representation)
    logits = layers.Dropout(0.5)(logits)
    logits = layers.Dense(mlp_head_units[1], activation=tf.nn.gelu)(logits)
    logits = layers.Dropout(0.5)(logits)
    logits = layers.Dense(num_classes, activation=tf.nn.softmax)(logits)

    # Create the Keras model.
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model

# Load the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

vit_model = create_vit_architecture()
vit_model.summary()

vit_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

vit_model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=16,
    validation_data=(x_test, y_test)
)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = vit_model.evaluate(x_test, y_test, batch_size=16)
print("test loss, test acc:", results)

#predict
print("predict")
results = vit_model.predict(x_test, batch_size=16)
print("predict:", results[0])
ans = np.argmax(results[0])
print("ans:", ans)
print("y_test:", y_test[0])
print("y_test:", np.argmax(y_test[0]))
