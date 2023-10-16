import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import numpy as np
import json
from sklearn.utils.class_weight import compute_class_weight

# tf random seed
tf.random.set_seed(42)

# load hyperparameters
with open('./jsons/parameters.json', 'r') as f:
    params = json.load(f)['transformer_hyperparameters']


class CustomMultiHeadAttentionLayer(layers.Layer):
    """
    A custom multi-head attention layer implementing residual connections and normalization.
    
    Methods
    -------
    call(inputs: tf.Tensor, training: bool=False) -> tf.Tensor:
        Forward pass through the attention mechanism.
    """

    def __init__(self, num_heads):
        """
        Initialize the CustomMultiHeadAttentionLayer instance.

        Parameters
        ----------
        num_heads : int
            The number of heads in the multi-head attention models.
        """
        super(CustomMultiHeadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        
    def build(self, input_shape):
        """
        Build the layer based on the input shape.

        Parameters
        ----------
        input_shape : tf.TensorShape
            The shape of the input tensor to the layer.
        """
        # Define a MultiHeadAttention layer
        self.mha = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=input_shape[2])

        # Define BatchNormalization layers
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        
        # Define a Dense layer
        self.dense = layers.Dense(input_shape[2])

        # Define an ELU activation function layer
        self.elu = layers.ELU()

        super().build(input_shape)
        
    def call(self, inputs, training=False):
        """
        Implement a forward pass through the multi-head attention mechanism.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor to the layer.
        training : bool, optional
            Whether the layer should behave in training mode or in inference mode. Defaults to False.
            
        Returns
        -------
        out : tf.Tensor
            Output tensor after passing through the attention mechanism.
        """
        # Multi-Head Attention with residual connection and Batch Normalization
        attn_output = self.mha(
            inputs,
            inputs,
            training=training
        )
        attn_output = self.bn1(inputs + attn_output, training=training)

        # Dense layer with residual connection and Batch Normalization
        dense_output = self.dense(attn_output, training=training)
        dense_output = self.bn2(attn_output + dense_output, training=training)
        out = self.elu(dense_output, training=training)

        return out

    
class CustomDenseLayer(layers.Layer):
    """
    A custom dense layer implementing dense connections, batch normalization, dropout, 
    and ELU activation.

    Methods
    -------
    call(inputs: tf.Tensor, training: bool=False) -> tf.Tensor:
        Forward pass through the dense mechanism.
    """

    def __init__(self, units, dropout_rate=0.1):
        """
        Initialize the CustomDenseLayer instance.

        Parameters
        ----------
        units : int
            The number of output nodes.
        dropout_rate : float, optional
            Fraction of the input units to drop. Defaults to 0.1.
        """
        super(CustomDenseLayer, self).__init__()
        self.units = units
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        """
        Build the layer based on the input shape.

        Parameters
        ----------
        input_shape : tf.TensorShape
            The shape of the input tensor to the layer.
        """
        # Define a Dense layer
        self.dense = layers.Dense(self.units)
        
        # Define a BatchNormalization layer
        self.bn = layers.BatchNormalization()
        
        # Define a Dropout layer
        self.dropout = layers.Dropout(self.dropout_rate)
        
        # Define an ELU activation function layer
        self.elu = layers.ELU()

        super().build(input_shape)

    def call(self, inputs, training=False):
        """
        Implement a forward pass through the dense mechanism.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor to the layer.
        training : bool, optional
            Whether the layer should behave in training mode or in inference mode. Defaults to False.

        Returns
        -------
        out : tf.Tensor
            Output tensor after passing through the dense mechanism.
        """
        # Dense layer
        x = self.dense(inputs, training=training)
        
        # Batch Normalization
        x = self.bn(x, training=training)
        
        # Dropout for regularization
        x = self.dropout(x, training=training)
        
        # ELU activation function
        out = self.elu(x, training=training)
        
        return out


class TransformerModel():
    """
    Implementation of a Transformer Model for some classification task.

    Attributes
    ----------
    num_classes : int
        Number of output classes.
    model : tf.keras.Model
        Instantiated and compiled Keras model.

    Methods
    -------
    build_model() -> tf.keras.Model:
        Build and return the transformer model.
    compute_class_weights(y: np.ndarray) -> np.ndarray:
        Compute and set class weights based on label distribution.
    train(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, path: str):
        Train the model using the provided data and parameters.
    load(path: str):
        Load model weights from the specified path.
    save(path: str):
        Save model weights to the specified path.
    predict(X_test: np.ndarray) -> np.ndarray:
        Return model predictions for the provided input data.
    """

    def __init__(self):
        """
        Initialize the TransformerModel instance.
        """
        self.num_classes = 4
        self.model = self.build_model()


    def build_model(self):
        """
        Build and return the transformer model.

        Returns
        -------
        tf.keras.Model
            A compiled Keras model.
        """

        inputs = layers.Input(shape=(params["frame_window"], params["input_shape"]))

        # transforer blocks
        x = inputs
        for _ in range(params["num_transformer_blocks"]):
            x = CustomMultiHeadAttentionLayer(params["num_heads"])(x)

        # dense layers before flattening the time dimension
        for units in params["pre_dense_layers"]:
            x = CustomDenseLayer(units, dropout_rate=params["dropout"])(x)
            x = CustomDenseLayer(units, dropout_rate=params["dropout"])(x)

        x = layers.Flatten()(x)

        # final dense layers
        for units in params["dense_layers"]:
            x = CustomDenseLayer(units, dropout_rate=params["dropout"])(x)
            x = CustomDenseLayer(units, dropout_rate=params["dropout"])(x)

        # softmax output
        output = layers.Dense(self.num_classes, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=output)

        return model


    def compute_class_weights(self, y):
        """
        Compute and set class weights based on label distribution.

        Parameters
        ----------
        y : np.ndarray
            Array of labels.

        Returns
        -------
        np.ndarray
            Computed class weights.
        """
        return compute_class_weight(class_weight='balanced',
                                             classes=np.arange(self.num_classes),
                                             y=y)


    def class_weighted_cross_entropy_func(self):
        """
        Generate a function for class-weighted categorical crossentropy loss.

        Returns
        -------
        function
            A function that computes the class-weighted categorical crossentropy 
            between true and predicted labels.
        """
        def class_weighted_ce(y_true, y_pred):
            """
            Compute the class-weighted categorical crossentropy.

            Parameters
            ----------
            y_true : tf.Tensor
                True labels, expected to be a one-hot encoded tensor.
            y_pred : tf.Tensor
                Predicted labels/probabilities.

            Returns
            -------
            tf.Tensor
                The computed class-weighted categorical crossentropy loss.
            """
            ce = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
            weights = tf.reduce_sum(y_true * self.class_weights_val, axis=-1)
            weights = weights / tf.reduce_sum(weights, axis=-1)[..., None]
            return tf.reduce_sum(weights * ce)
        return class_weighted_ce


    def train(self, X, y, X_val, y_val, path):
        """
        Train the model using the provided data and parameters.

        Parameters
        ----------
        X : np.ndarray
            Training input data.
        y : np.ndarray
            Training labels.
        X_val : np.ndarray
            Validation input data.
        y_val : np.ndarray
            Validation labels.
        path : str
            Path where model weights will be stored.
        """

        # compute class weights
        self.class_weights = self.compute_class_weights(y)
        self.class_weights_val = self.compute_class_weights(y_val)

        # convert y to one-hot
        y = to_categorical(y, num_classes=self.num_classes)
        y_val = to_categorical(y_val, num_classes=self.num_classes)

        # compile the model with adam optimizer and focal loss
        self.model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=params['learning_rate']),
                          loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=self.class_weights),
                          metrics=['accuracy', self.class_weighted_cross_entropy_func()])
        
        # create a callback to save model at checkpoints
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=path,
            save_weights_only=True,
            monitor='val_class_weighted_ce',
            mode='min',
            save_best_only=True
        )
        
        # train the model
        self.model.fit(X, y, epochs=params['epochs'], batch_size=params['batch_size'], validation_data=(X_val, y_val), callbacks=[model_checkpoint_callback])

    def load(self, path):
        """
        Load model weights from the specified path.

        Parameters
        ----------
        path : str
            Path to the model weights.
        """
        self.model.load_weights(path)

    def save(self, path):
        """
        Save model weights to the specified path.

        Parameters
        ----------
        path : str
            Path where model weights will be stored.
        """
        self.model.save_weights(path)

    def predict(self, X_test):
        """
        Return model predictions for the provided input data.

        Parameters
        ----------
        X_test : np.ndarray
            Input data for predictions.

        Returns
        -------
        np.ndarray
            Model predictions.
        """
        return self.model.predict(X_test)