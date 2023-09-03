# Import the required libraries.

import random
import numpy as np
from sklearn.model_selection import train_test_split

from video_classification import test_video
from frame_extraction import create_dataset, SEQUENCE_LENGTH, IMAGE_WIDTH, IMAGE_HEIGHT
from models import *

# Set Numpy, Python, and Tensorflow seeds to get consistent results on every execution.
seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)


if __name__ == '__main__':
    epochs = 70
    features, labels = create_dataset('videos')

    features_train, features_test, labels_train, labels_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        shuffle=True,
        random_state=seed_constant
    )

    conv_lstm_model = create_conv_lstm_model(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH)
    cnn_model = create_cnn_model(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH)
    two_streams_cnn_model = create_two_streams_cnn_model(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH)

    source_path = 'Tests/source'
    output_path = 'results/videos'

    train_and_capture_history(cnn_model, features_train, labels_train, features_test, labels_test, epochs=epochs, model_name='cnn_model')
    test_video(cnn_model, source_path, output_path, 'violent-1.mp4', 'cnn_model_violent-1.mp4')
    test_video(cnn_model, source_path, output_path, 'non-violent-1.mp4', 'cnn_model_non-violent-1.mp4')

    # train_and_capture_history(two_streams_cnn_model, features_train, labels_train, features_test, labels_test, epochs=epochs, model_name='two_streams_cnn_model')
    # test_video(two_streams_cnn_model, source_path, output_path, 'violent-1.mp4', 'two_streams_cnn_model_violent-1.mp4')
    # test_video(two_streams_cnn_model, source_path, output_path, 'non-violent-1.mp4', 'two_streams_cnn_model_non-violent-1.mp4')
    #
    # train_and_capture_history(conv_lstm_model, features_train, labels_train, features_test, labels_test, epochs=epochs, model_name='conv_lstm_model')
    # test_video(conv_lstm_model, source_path, output_path, 'violent-1.mp4', 'conv_lstm_model_violent-1.mp4')
    # test_video(conv_lstm_model, source_path, output_path, 'non-violent-1.mp4', 'conv_lstm_model_non-violent-1.mp4')

    # vgg16_model = use_vgg16(epochs)
    # test_video(vgg16_model, source_path, output_path, 'violent-1.mp4', 'vgg16_model_violent-1.mp4')
    # test_video(vgg16_model, source_path, output_path, 'non-violent-1.mp4', 'vgg16_model_non-violent-1.mp4')


