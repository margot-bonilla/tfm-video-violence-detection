import os
from collections import deque
from frame_extraction import IMAGE_WIDTH, IMAGE_HEIGHT, SEQUENCE_LENGTH
import numpy as np

import cv2


def move_videos_files(source, destination, folder_name):
    for f in os.listdir(source):
        src_path = os.path.join(source, f)
        dst_path = os.path.join(destination, f'{folder_name}_{f}')
        os.rename(src_path, dst_path)


def trim_video_names(path):
    for f in os.listdir(path):
        src_path = os.path.join(path, f)
        dst_path = os.path.join(path, f.strip())
        os.rename(src_path, dst_path)


def move_movies_videos():
    """
    This code is to move all the videos in the folders "violent" and "non-violent"
    :return:
    """

    """
    # movies dataset are classified as
    fights
        | - 1.avi
        | - 2.avi
        | - ...
    noFights
        | - 1.avi
    """
    move_videos_files('Datasets/Movies/fights', 'videos/violent', 'movies')
    move_videos_files('Datasets/Movies/noFights', 'videos/non-violent', 'movies')

    """
    # violence dataset are classified as
    violent
        | - cam1
            | - 1.mp4
            | - 2.mp4
            | - ...
        | - cam1
    non-violent
        | - cam1
            | - 1.mp4
        | - cam2
            | - 1.mp4
    """
    move_videos_files('Datasets/violence-detection-dataset/violent/cam1', 'videos/violent', 'cam1')
    move_videos_files('Datasets/violence-detection-dataset/violent/cam2', 'videos/violent', 'cam2')
    move_videos_files('Datasets/violence-detection-dataset/non-violent/cam1', 'videos/non-violent', 'cam1')
    move_videos_files('Datasets/violence-detection-dataset/non-violent/cam2', 'videos/non-violent', 'cam2')

    # for Hockey dataset we need to redistribute the videos first in fight noFight and then move to the folders
    hockey_path = 'Datasets/HockeyFights'
    for v in os.listdir(hockey_path):
        src_path = os.path.join(hockey_path, v)
        if os.path.isfile(src_path):
            dst_folder = 'noFights' if v.startswith('no') else 'fights'
            dst_path = os.path.join(hockey_path, dst_folder, v)
            os.rename(src_path, dst_path)

    move_videos_files('Datasets/HockeyFights/fights', 'videos/violent', 'hockey')
    move_videos_files('Datasets/HockeyFights/noFights', 'videos/non-violent', 'hockey')

    # remove all spaces on the names
    trim_video_names('videos/violent')
    trim_video_names('videos/non-violent')


CLASSES_LIST = ['no violent', 'violent']


def predict_on_video(model, video_file_path, output_file_path, SEQUENCE_LENGTH):
    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the VideoWriter Object to store the output video in the disk.
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    # Declare a queue to store video frames.
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Iterate until the video is accessed successfully.
    while video_reader.isOpened():

        # Read the frame.
        ok, frame = video_reader.read()

        # Check if frame is not read properly then break the loop.
        if not ok:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list.
        frames_queue.append(normalized_frame)

        # Check if the number of frames in the queue are equal to the fixed sequence length.
        if len(frames_queue) == SEQUENCE_LENGTH:
            # Pass the normalized frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]

            # Get the index of class with the highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)

            # Get the class name using the retrieved index.
            predicted_class_name = CLASSES_LIST[predicted_label]

        # Write predicted class name on top of the frame.
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write The frame into the disk using the VideoWriter Object.
        video_writer.write(frame)

    # Release the VideoCapture and VideoWriter objects.
    video_reader.release()
    video_writer.release()


def test_video(model, input_video_file_path, output_video_file_path, input_file_name, output_file_name):
    # # Make the Output directory if it does not exist
    # test_videos_directory = '/content/drive/MyDrive/TFM/VideoResults/test_videos'
    # os.makedirs(test_videos_directory, exist_ok=True)
    # video_title = 'violent-1'
    # input_video_file_path = f'{test_videos_directory}/{video_title}.mp4'
    # # Construct the output video path.
    # output_video_file_path = f'{test_videos_directory}/{video_title}-Output-SeqLen{SEQUENCE_LENGTH}.mp4'
    os.makedirs(output_video_file_path, exist_ok=True)

    # Perform Action Recognition on the Test Video.
    predict_on_video(model, f'{input_video_file_path}/{input_file_name}', f'{output_video_file_path}/{output_file_name}', SEQUENCE_LENGTH)
