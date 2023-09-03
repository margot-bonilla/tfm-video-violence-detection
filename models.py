import tensorflow as tf

import datetime as dt
import matplotlib.pyplot as plt

from keras.callbacks import CSVLogger
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

metrics = ["accuracy"]


def create_conv_lstm_model(sequence_length, image_height, image_width):
    # Define the input shape for ConvLSTM2D
    input_shape = (sequence_length, image_height, image_width, 3)

    # Create an input layer
    input_layer = Input(shape=input_shape)

    # Define the ConvLSTM2D layer
    conv_lstm = ConvLSTM2D(64, (3, 3), activation='relu', padding='same', return_sequences=False)(input_layer)

    # Flatten the output
    flatten = Flatten()(conv_lstm)

    # Add fully connected layers
    dense1 = Dense(128, activation='relu')(flatten)
    dropout = Dropout(0.5)(dense1)

    # Output layer for binary classification
    output_layer = Dense(1, activation='sigmoid')(dropout)

    # Create the ConvLSTM2D model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

    # Print a summary of the model architecture
    model.summary()
    plot_model(model, to_file='results/models/convlstm_model.png', show_shapes=True, show_layer_names=True)

    return model


def create_cnn_model(sequence_length=10, image_height=64, image_width=64):
    """
    # We define a sequential model.
    # We add two 3D convolutional layers with max-pooling layers to extract spatial features from the video frames.
    # The Flatten layer is used to transform the 3D feature maps into a 1D vector.
    # We add a fully connected layer with ReLU activation and a dropout layer for regularization.
    # Finally, we add an output layer with a single neuron and sigmoid activation for binary classification.
    :param sequence_length: number of frames
    :param image_height: image height
    :param image_width: image width
    :return:
    """
    # Define the model
    model = Sequential()

    # Convolutional layers
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu',
                     input_shape=(sequence_length, image_height, image_width, 3)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # Flatten the 3D features to feed into a fully connected layer
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=metrics)

    model.summary()
    plot_model(model, to_file='results/models/cnn_model.png', show_shapes=True, show_layer_names=True)

    return model


def create_rnn_model(sequence_length, image_height, image_width):
    # Define the input shape for the LSTM model
    input_shape = (30, image_height * image_width)  # (time_steps, flattened_features)

    # Create an input layer
    input_layer = Input(shape=input_shape)

    # Reshape the input to (batch_size, time_steps, features)
    reshaped_input = Reshape(target_shape=(sequence_length, image_height * image_width))(input_layer)

    # Define the 3D RNN model with LSTM layers
    x = LSTM(64, return_sequences=True)(reshaped_input)
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(64)(x)

    # Add fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Output layer for binary classification
    output_layer = Dense(1, activation='sigmoid')(x)

    # Create the 3D RNN model
    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=metrics)
    model.summary()
    plot_model(model, to_file='results/models/rnn_model.png', show_shapes=True, show_layer_names=True)

    return model


def create_vgg16_model(sequence_length, image_height, image_width):
    """
    Create a VGG16-based model for video classification.

    Parameters:
    - sequence_length: Number of frames in the video sequence
    - image_height: Height of each frame
    - image_width: Width of each frame

    Returns:
    - model: VGG16-based model
    """
    # Define the input shape for a single frame (height, width, channels)
    input_shape = (image_height, image_width, 3)

    # Create an input layer with the specified input shape
    input_layer = Input(shape=input_shape)

    # Repeat the input for each frame in the sequence
    repeated_input = Lambda(lambda x: tf.stack([x] * sequence_length, axis=1))(input_layer)

    # Load the VGG16 model pre-trained on ImageNet data (excluding the top layers)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Apply the VGG16 model to each frame in the sequence
    frame_outputs = TimeDistributed(base_model)(repeated_input)

    # Flatten the frame outputs
    x = TimeDistributed(Flatten())(frame_outputs)

    # Add fully connected layers
    x = Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)  # Use tf.keras.layers.Dropout for consistency
    output_layer = Dense(1, activation='sigmoid')(x)

    # Create the VGG16-based model for video classification
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=metrics)

    # Print a summary of the model architecture
    model.summary()

    return model


# Define a 3D ResNet architecture
def create_3d_resnet_model(sequence_length, image_height, image_width):
    """
    # We load the pre-trained 3D ResNet50 model (ResNet50V2) pre-trained on the ImageNet dataset using ResNet50
    from tensorflow.keras.applications. We specify include_top=False to exclude the fully connected layers of the model.
    # We freeze the layers in the base model to prevent them from being updated during training.
    # We add custom layers on top of the base model to adapt it for video classification.
    This includes a global average pooling layer, a dense hidden layer with ReLU activation,
    and an output layer with a sigmoid activation for binary classification.
    # The final model is created with the custom layers added on top of the base model.
    # The model is compiled with an appropriate optimizer, loss function, and metrics.
    # This allows you to use the pre-trained 3D ResNet50 model as a feature extractor for video frames and
    then fine-tune it on your specific video classification task.
    :param sequence_length:
    :param image_height:
    :param image_width:
    :return:
    """
    # Load the pre-trained 3D ResNet50 model without the top classification layers
    base_model = ResNet50(weights='imagenet', include_top=False,
                          input_shape=(sequence_length, image_height, image_width, 3))

    # Freeze the layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers for video classification
    x = base_model.output
    x = GlobalAveragePooling3D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)  # Output layer for binary classification

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=metrics)
    model.summary()
    plot_model(model, to_file='results/3d_resnet_model.png', show_shapes=True, show_layer_names=True)

    return model


def create_two_streams_cnn_model(sequence_length, image_height, image_width):
    # Define the input shape for both spatial and temporal streams
    input_shape = (sequence_length, image_height, image_width, 3)

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the spatial stream
    spatial_stream = Conv3D(64, (3, 3, 3), activation='relu')(input_layer)
    spatial_stream = MaxPooling3D(pool_size=(2, 2, 2))(spatial_stream)
    spatial_stream = Conv3D(128, (3, 3, 3), activation='relu')(spatial_stream)
    spatial_stream = MaxPooling3D(pool_size=(2, 2, 2))(spatial_stream)
    spatial_stream = Flatten()(spatial_stream)
    spatial_stream = Dense(128, activation='relu')(spatial_stream)

    # Define the temporal stream
    temporal_stream = Conv3D(64, (3, 3, 3), activation='relu')(input_layer)
    temporal_stream = MaxPooling3D(pool_size=(2, 2, 2))(temporal_stream)
    temporal_stream = Conv3D(128, (3, 3, 3), activation='relu')(temporal_stream)
    temporal_stream = MaxPooling3D(pool_size=(2, 2, 2))(temporal_stream)
    temporal_stream = Flatten()(temporal_stream)
    temporal_stream = Dense(128, activation='relu')(temporal_stream)

    # Concatenate the outputs of the two streams
    merged = Concatenate()([spatial_stream, temporal_stream])

    # Add more fully connected layers if needed
    merged = Dense(128, activation='relu')(merged)

    # Output layer for binary classification
    output = Dense(1, activation='sigmoid')(merged)

    # Create the two-stream model
    model = Model(inputs=input_layer, outputs=output)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
    model.summary()
    plot_model(model, to_file='results/models/two_streams_cnn_model.png', show_shapes=True, show_layer_names=True)

    return model


def use_vgg16(epochs):
    # Load the pre-trained VGG16 model without the top classification layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add custom classification layers on top of VGG16
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Data generators for training and validation
    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        'frames',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        'frames',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=epochs
    )

    # Evaluate the model on test data
    test_generator = test_datagen.flow_from_directory(
        'frames',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    # Save the model
    model.save('results/models/vgg16_violence_detection_model.keras')

    plot_metric(history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy VGG16')
    plot_metric(history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss VGG16')

    return model


def train_and_capture_history(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32, model_name=''):
    csv_logger = CSVLogger(f'results/logger/training_{model_name}.log', separator=',', append=False)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)
    checkpoint_callback = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

    model_training_history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping_callback, checkpoint_callback, csv_logger]
    )

    model_evaluation_history = model.evaluate(X_test, y_test)

    # Get the loss and accuracy from model_evaluation_history.
    model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history

    # Define the string date format.
    # Get the current Date and Time in a DateTime Object.
    # Convert the DateTime object to string according to the style mentioned in date_time_format string.
    date_time_format = '%Y_%m_%d__%H_%M_%S'
    current_date_time_dt = dt.datetime.now()
    current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)

    # Define a useful name for our model to make it easy for us while navigating through multiple saved models.
    model_file_name = f'{model_name}___Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.keras'

    # Save your Model.
    model.save(f'results/models/{model_file_name}', overwrite=True)

    plot_metric(model_training_history, 'accuracy', 'val_accuracy', f'Total Accuracy vs Total Validation Accuracy {model_name}')
    plot_metric(model_training_history, 'loss', 'val_loss', f'Total Loss vs Total Validation Loss {model_name}')


def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
    # Get metric values using metric names as identifiers.
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]

    # Construct a range object which will be used as x-axis (horizontal plane) of the graph.
    epochs = range(len(metric_value_1))

    # Plot the Graph.
    plt.plot(epochs, metric_value_1, 'blue', label=metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label=metric_name_2)

    # Add title to the plot.
    plt.title(str(plot_name))

    # Add legend to the plot.
    plt.legend()
    plt.savefig(f'results/performance/{plot_name}.png', overwrite=True)
