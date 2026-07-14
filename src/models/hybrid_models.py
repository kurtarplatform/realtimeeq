"""
Hybrid CNN-RNN Architectures — Edge-AI Based Seismic Detection Framework
=========================================================================
CNN feature extractors + sequential models (LSTM/BiLSTM/GRU) for
temporal seismic signal modelling.

Architectures:
    1. CNN-LSTM: Conv1D blocks → LSTM → Dense (baseline hybrid)
    2. CNN-BiLSTM-Attention: Conv1D → Bidirectional LSTM → Attention → Dense
    3. CNN-GRU: Conv1D blocks → GRU → Dense (lightweight edge-AI variant)

Design constraints:
    - Input: (200, 3) — 2s window @ 100 Hz, 3-axis accelerometer
    - Output: sigmoid → P(earthquake)
    - Parameter budget: 50K – 200K (TFLite compatible)

References:
    Mousavi et al. (2020) "Earthquake transformer" — Attention for seismology
    Kong et al. (2019) "MyShake" — Mobile seismic detection
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers


# ─── Shared CNN Feature Extractor ────────────────────────────────────────────

def _cnn_feature_extractor(
    inputs,
    filters=(32, 64),
    kernel_sizes=(7, 5),
    pool_sizes=(2, 2),
    dropout_rate=0.3,
    l2_reg=1e-4,
    prefix="cnn",
):
    """
    Shared Conv1D feature extractor used by all hybrid architectures.
    Returns temporal feature maps (time_reduced, filters[-1]).
    """
    x = inputs
    for i, (f, k, p) in enumerate(zip(filters, kernel_sizes, pool_sizes)):
        x = layers.Conv1D(
            f, k, padding="same",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{prefix}_conv{i+1}",
        )(x)
        x = layers.BatchNormalization(name=f"{prefix}_bn{i+1}")(x)
        x = layers.Activation("relu", name=f"{prefix}_relu{i+1}")(x)
        x = layers.MaxPooling1D(p, name=f"{prefix}_pool{i+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"{prefix}_drop{i+1}")(x)
    return x


# ─── Classification Head ────────────────────────────────────────────────────

def _classification_head(x, dense_units=32, dropout_rate=0.4, l2_reg=1e-4, prefix="head"):
    """Shared Dense → Dropout → Sigmoid head."""
    x = layers.Dense(
        dense_units, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{prefix}_dense",
    )(x)
    x = layers.Dropout(dropout_rate, name=f"{prefix}_drop")(x)
    x = layers.Dense(1, activation="sigmoid", name="eq_probability")(x)
    return x


# ═════════════════════════════════════════════════════════════════════════════
# Architecture 1: CNN-LSTM
# ═════════════════════════════════════════════════════════════════════════════

def build_cnn_lstm(
    input_shape=(200, 3),
    cnn_filters=(32, 64),
    cnn_kernels=(7, 5),
    cnn_pools=(2, 2),
    lstm_units=48,
    dense_units=32,
    dropout_rate=0.3,
    recurrent_dropout=0.2,
    l2_reg=1e-4,
    name="CNN_LSTM",
):
    """
    CNN-LSTM hybrid: Conv1D feature extraction → LSTM temporal modelling.

    Architecture:
        Input(200,3) → [Conv1D→BN→ReLU→Pool→Drop]×2 → LSTM → Dense → sigmoid

    The CNN blocks reduce temporal resolution while extracting local features.
    The LSTM then models sequential dependencies across the reduced time steps.

    Parameters
    ----------
    input_shape : tuple
        (time_steps, channels) = (200, 3).
    cnn_filters : tuple
        Filters per Conv1D block.
    lstm_units : int
        LSTM hidden units.
    dense_units : int
        Classification head units.
    dropout_rate : float
        Spatial dropout rate.
    recurrent_dropout : float
        LSTM recurrent dropout.

    Returns
    -------
    keras.Model
    """
    inputs = layers.Input(shape=input_shape, name="accelerometer_input")

    # CNN feature extractor
    x = _cnn_feature_extractor(
        inputs, cnn_filters, cnn_kernels, cnn_pools,
        dropout_rate, l2_reg, prefix="cnn"
    )

    # LSTM temporal modelling
    x = layers.LSTM(
        lstm_units,
        return_sequences=False,
        dropout=dropout_rate,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=regularizers.l2(l2_reg),
        name="lstm",
    )(x)

    # Classification head
    outputs = _classification_head(x, dense_units, dropout_rate, l2_reg)

    return Model(inputs=inputs, outputs=outputs, name=name)


# ═════════════════════════════════════════════════════════════════════════════
# Architecture 2: CNN-BiLSTM-Attention
# ═════════════════════════════════════════════════════════════════════════════

class BahdanauAttention(layers.Layer):
    """
    Additive (Bahdanau) attention mechanism.

    Learns to weight each time step of the BiLSTM output sequence,
    enabling the model to focus on the most discriminative temporal
    segments (e.g., P-wave arrival).

    Reference: Bahdanau et al. (2015) "Neural Machine Translation by
    Jointly Learning to Align and Translate"
    """

    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_W", shape=(input_shape[-1], self.units),
            initializer="glorot_uniform", trainable=True,
        )
        self.b = self.add_weight(
            name="att_b", shape=(self.units,),
            initializer="zeros", trainable=True,
        )
        self.v = self.add_weight(
            name="att_v", shape=(self.units,),
            initializer="glorot_uniform", trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, time_steps, features)
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)  # (B, T, units)
        attention_weights = tf.nn.softmax(
            tf.reduce_sum(score * self.v, axis=-1, keepdims=True), axis=1
        )  # (B, T, 1)
        context = tf.reduce_sum(inputs * attention_weights, axis=1)  # (B, features)
        return context, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_cnn_bilstm_attention(
    input_shape=(200, 3),
    cnn_filters=(32, 64),
    cnn_kernels=(7, 5),
    cnn_pools=(2, 2),
    lstm_units=32,
    attention_units=24,
    dense_units=32,
    dropout_rate=0.3,
    recurrent_dropout=0.2,
    l2_reg=1e-4,
    name="CNN_BiLSTM_Attn",
):
    """
    CNN-BiLSTM with Bahdanau Attention.

    Architecture:
        Input(200,3) → [Conv1D→BN→ReLU→Pool→Drop]×2
            → Bidirectional(LSTM, return_sequences=True)
            → BahdanauAttention → Dense → sigmoid

    The attention mechanism learns temporal saliency weights, enabling
    the model to focus on P-wave onset regions. This is the primary
    novelty architecture of the proposed edge-AI framework.

    Returns
    -------
    keras.Model
    """
    inputs = layers.Input(shape=input_shape, name="accelerometer_input")

    x = _cnn_feature_extractor(
        inputs, cnn_filters, cnn_kernels, cnn_pools,
        dropout_rate, l2_reg, prefix="cnn"
    )

    # Bidirectional LSTM (return full sequence for attention)
    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizers.l2(l2_reg),
        ),
        name="bilstm",
    )(x)

    # Attention
    x, _ = BahdanauAttention(units=attention_units, name="attention")(x)

    # Classification head
    outputs = _classification_head(x, dense_units, dropout_rate, l2_reg)

    return Model(inputs=inputs, outputs=outputs, name=name)


# ═════════════════════════════════════════════════════════════════════════════
# Architecture 3: CNN-GRU (Lightweight Edge-AI)
# ═════════════════════════════════════════════════════════════════════════════

def build_cnn_gru(
    input_shape=(200, 3),
    cnn_filters=(32, 48),
    cnn_kernels=(7, 5),
    cnn_pools=(2, 2),
    gru_units=32,
    dense_units=24,
    dropout_rate=0.3,
    recurrent_dropout=0.2,
    l2_reg=1e-4,
    name="CNN_GRU",
):
    """
    CNN-GRU lightweight hybrid — optimised for edge-AI deployment.

    GRU has ~25% fewer parameters than LSTM (2 gates vs 3) while
    maintaining comparable sequence modelling performance. This makes
    it the preferred architecture for battery-constrained mobile devices.

    Architecture:
        Input(200,3) → [Conv1D→BN→ReLU→Pool→Drop]×2 → GRU → Dense → sigmoid

    Returns
    -------
    keras.Model
    """
    inputs = layers.Input(shape=input_shape, name="accelerometer_input")

    x = _cnn_feature_extractor(
        inputs, cnn_filters, cnn_kernels, cnn_pools,
        dropout_rate, l2_reg, prefix="cnn"
    )

    # GRU temporal modelling
    x = layers.GRU(
        gru_units,
        return_sequences=False,
        dropout=dropout_rate,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=regularizers.l2(l2_reg),
        name="gru",
    )(x)

    # Classification head
    outputs = _classification_head(x, dense_units, dropout_rate, l2_reg)

    return Model(inputs=inputs, outputs=outputs, name=name)


# ─── Utilities ───────────────────────────────────────────────────────────────

def compile_hybrid(model, learning_rate=1e-3):
    """Compile hybrid model with standard metrics."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc", curve="ROC"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def get_model_stats(model):
    """Return model summary dict."""
    trainable = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable = sum(
        tf.keras.backend.count_params(w) for w in model.non_trainable_weights
    )
    return {
        "name": model.name,
        "total_params": trainable + non_trainable,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "n_layers": len(model.layers),
    }


def compare_architectures(input_shape=(200, 3)):
    """Build all 3 architectures and print comparison table."""
    builders = [
        ("CNN-LSTM", build_cnn_lstm),
        ("CNN-BiLSTM-Attn", build_cnn_bilstm_attention),
        ("CNN-GRU", build_cnn_gru),
    ]
    print(f"\n{'='*65}")
    print(f"  {'Architecture':<22} {'Trainable':>12} {'Total':>12} {'Layers':>8}")
    print(f"  {'-'*54}")
    for name, builder in builders:
        m = builder(input_shape=input_shape)
        s = get_model_stats(m)
        budget = "✅" if 50_000 <= s['trainable_params'] <= 200_000 else "⚠️"
        print(f"  {name:<22} {s['trainable_params']:>10,}  {s['total_params']:>10,}  {s['n_layers']:>6}  {budget}")
        del m
    print(f"{'='*65}")
    print(f"  Budget target: 50K – 200K trainable parameters")


if __name__ == "__main__":
    compare_architectures()
