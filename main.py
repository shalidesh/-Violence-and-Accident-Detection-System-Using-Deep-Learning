
import cv2

src_dir = "Dataset_V_6.mp4"
dst_dir = "output.avi"

video_reader = cv2.VideoCapture(src_dir)

# Get the width and height of the video.
original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize the VideoWriter Object to store the output video in the disk.
video_writer = cv2.VideoWriter(dst_dir, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
                                video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))


success, frame = video_reader.read()
while success:
    video_writer.write(frame)
    success, frame = video_reader.read()