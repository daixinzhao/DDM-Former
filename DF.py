import argparse
import numpy as np
import tensorflow as tf
import os

from tensorflow.keras import layers
import tensorflow_addons as tfa

parser = argparse.ArgumentParser(description="DF")
parser.add_argument('--batch_size', type=int, default=512, help='batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate for training')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay for training')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs for training')
parser.add_argument('--patience', type=int, default=6, help='early-stopping patience')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for the train loader')
parser.add_argument('--patch_size', type=int, default=1, help='DF patch size')
parser.add_argument('--num_patches', type=int, default=187, help='DF patch number')
parser.add_argument('--projection_dim', type=int, default=187, help='linear projection dimension')
parser.add_argument('--num_heads', type=int, default=11, help='number of heads')
parser.add_argument('--transformer_layers', type=int, default=2, help='number of transformer layers')
parser.add_argument('--mlp_head_units_0', type=int, default=512, help='mlp head 0 units')
parser.add_argument('--mlp_head_units_1', type=int, default=128, help='mlp head 1 units')
args = parser.parse_args()

# set random seed
np.random.seed(0)
tf.random.set_seed(0)

# load dataset
x_train = # training DDMs
y_train = # training label
print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")

x_valid = # validation DDMs
y_valid = # validation label
print(f"x_valid shape: {x_valid.shape} - y_valid shape: {y_valid.shape}")

# model save directory
weights_dir = 'logs_DF/weights/'
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

board_dir = 'logs_DF/events/'
if not os.path.exists(board_dir):
    os.makedirs(board_dir)

# tr block setup
transformer_units = [args.projection_dim * 4, args.projection_dim]
mlp_head_units = [args.mlp_head_units_0, args.mlp_head_units_1]
initializer = tf.keras.initializers.GlorotUniform()

# mlp sublayer
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu, kernel_initializer=initializer)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# pixel-wise tokenization 
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
        })
        return config

# DDA token
class DDAToken(layers.Layer): 
    def __init__(self, patch_size):
        super(DDAToken, self).__init__()
        dda_init = tf.zeros_initializer()
        self.hidden_size = patch_size * patch_size * 4
        self.dda = tf.Variable(
            name="dda",
            initial_value=dda_init(shape=(1, 1, self.hidden_size), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        dda_broadcasted = tf.cast(
            tf.broadcast_to(self.dda, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        concat = tf.concat([dda_broadcasted, inputs], 1)
        return concat

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
        })
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches+1
        self.projection = layers.Dense(units=projection_dim, kernel_initializer=initializer)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_pathces": self.num_patches,
            "projection_dim": self.projection,
        })
        return config

# msa sublayer
@tf.keras.utils.register_keras_serializable()
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, *args, num_heads, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedded dimension not divisible by number of heads"
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query_dense = tf.keras.layers.Dense(hidden_size, name="query")
        self.key_dense = tf.keras.layers.Dense(hidden_size, name="key")
        self.value_dense = tf.keras.layers.Dense(hidden_size, name="value")
        self.combine_heads = tf.keras.layers.Dense(hidden_size, name="out")

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention)
        return output, weights

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        return config

    @classmethod
    def from_config(dda, config):
        return dda(**config)

def ddm_former():
    inputs = layers.Input(shape=(17,11,4))
    # create patches
    patches = Patches(args.patch_size)(inputs)
    dda = DDAToken(args.patch_size)(patches)
    # encode patches
    encoded_patches = PatchEncoder(args.num_patches, args.projection_dim)(dda)
    # create tr block
    for _ in range(args.transformer_layers):
        # LN
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # msa
        attention_output, _ = MultiHeadSelfAttention(num_heads=args.num_heads)(x1)
        # skip connection
        x2 = layers.Add()([attention_output, encoded_patches])
        # LN
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # mlp
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # skip connection
        encoded_patches = layers.Add()([x3, x2])

    out_repre = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    out_repre = layers.Flatten()(out_repre)
    out_repre = layers.Dropout(0.5)(out_repre)
    # mlp
    out_repre = mlp(out_repre, hidden_units=mlp_head_units, dropout_rate=0.5)
    # ws
    ws = layers.Dense(1, kernel_initializer=initializer)(out_repre)
    # create model
    model = tf.keras.Model(inputs=inputs, outputs=ws)
    return model

print('---------Training----------')
df_model = ddm_former()
optimizer = tfa.optimizers.AdamW(learning_rate=args.learning_rate,
                                 weight_decay=args.weight_decay)
df_model.compile(optimizer=optimizer,
                loss="mse",
                metrics=[tf.keras.metrics.MeanSquaredError(name='MSE')])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=args.patience)
model_ckt = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(weights_dir, 'weights.h5'),
                            verbose=1,
                            save_best_only=True)
tfboard = tf.keras.callbacks.TensorBoard(log_dir=board_dir,
                        write_graph=True,
                        write_images=True)
df_model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
            callbacks=[model_ckt, tfboard, callback],
            validation_data=(x_valid, y_valid),
            shuffle=True, workers=args.num_workers)
df_model.save(os.path.join(weights_dir, 'model-DF.tf'))
print('---------Training Done---------')