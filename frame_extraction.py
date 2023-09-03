import cv2
import os
import numpy as np

# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

# Number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 30


def frames_extraction(video_path):
    # Declare a list to store video frames.
    frames_list = []

    # Read the Video File using the VideoCapture object.
    video_reader = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

    # Iterate through the Video Frames.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Reading the frame from the video.
        success, frame = video_reader.read()

        # Check if Video frame is not successfully read then break the loop
        if not success:
            break

        # Resize the Frame to fixed height and width.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Append the normalized frame into the frames list
        frames_list.append(normalized_frame)

    # Release the VideoCapture object.
    video_reader.release()

    # Return the frames list.
    return frames_list


def create_dataset(video_folder, test=False):
    # Declared Empty Lists to store the features, labels.
    features = []
    labels = []

    violent_videos_folder = f'{video_folder}/violent'
    non_violent_videos_folder = f'{video_folder}/non-violent'

    violent_videos = os.listdir(violent_videos_folder)
    non_violent_videos = os.listdir(non_violent_videos_folder)
    print(len(violent_videos) + len(non_violent_videos))

    if test:
        violent_videos = violent_videos[:10]
        non_violent_videos = non_violent_videos[:10]

    # This is to show the progress of the process
    total = len(violent_videos) + len(non_violent_videos)
    processed = 0

    # Iterate through all the files present in the files list.
    for non_violent_file_path in non_violent_videos:
        non_violent_frames = frames_extraction(os.path.join(non_violent_videos_folder, non_violent_file_path))

        if len(non_violent_frames) == SEQUENCE_LENGTH:
            # Append the data to their respective lists.
            features.append(non_violent_frames)
            labels.append(0)
            processed += 1

        print(f'Processed {processed} out of {total}, {round((processed / total) * 100, 2)}%')

    # Iterate through all the files present in the files list.
    for violent_file_path in violent_videos:
        violent_frames = frames_extraction(os.path.join(violent_videos_folder, violent_file_path))

        if len(violent_frames) == SEQUENCE_LENGTH:
            # Append the data to their respective lists.
            features.append(violent_frames)
            labels.append(1)
            processed += 1

        print(f'Processed {processed} out of {total}, {round((processed / total) * 100, 2)}%')

    # Converting the list to numpy arrays
    features = np.asarray(features)
    labels = np.array(labels)

    # Return the frames, class index, and video file path.
    return features, labels


def extract_frames_in_output(video_dir, output_dir, desired_height=244, desired_width=244):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Supported video extensions
    video_extensions = ['.mp4', '.avi', '.mov']  # Add more extensions as needed

    # Iterate through video files
    for video_file in os.listdir(video_dir):
        file_name, file_extension = os.path.splitext(video_file)
        if file_extension.lower() in video_extensions:
            video_path = os.path.join(video_dir, video_file)
            vidcap = cv2.VideoCapture(video_path)
            success, image = vidcap.read()
            count = 0
            while success:
                # Resize the frame to the desired size
                resized_frame = cv2.resize(image, (desired_width, desired_height))

                # Normalize pixel values to the range [0, 1]
                normalized_frame = resized_frame / 255.0

                frame_filename = f'{file_name}_frame_{count}.jpg'
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, normalized_frame)  # Save the normalized frame as an image
                success, image = vidcap.read()
                count += 1
