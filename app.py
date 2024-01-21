
# Import the required libraries.
from flask import Flask, render_template, url_for, request,flash,redirect,Response,send_from_directory
import os
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from werkzeug.utils import secure_filename

#for alert system
import requests
from datetime import datetime, timedelta
import time
import pytz
from os import environ



app = Flask(__name__)

app.config['UPLOAD_FOLDER']='static/assets/upload_vedios'



# Make the Output directory if it does not exist
test_videos_directory = 'static/assets/test_vedios'
os.makedirs(test_videos_directory, exist_ok = True)

LRCN_model=tf.keras.models.load_model('models\LRCN_model___Date_Time_2022_11_27__05_44_56___Loss_0.3180044889450073___Accuracy_0.8740000128746033.h5')
 

# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
 
# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 15
 
# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
CLASSES_LIST = ['NonViolence', 'Violence']


#Alert System Constants

IST = pytz.timezone('Asia/Kolkata') 
raw_TS = datetime.now(IST)
curr_date = raw_TS.strftime("%d-%m-%Y") #Current Date
curr_time = raw_TS.strftime("%H:%M:%S")  #Current time
telegram_auth_token = "5888406295:AAGSfKxmlwAsMZOBOr-bEArIHPIP2SSbFh8" # Authentication token provided by Telegram bot
telegram_group_id = "violence_detecter"      # Telegram group name




def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH):
    
    
    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize the VideoWriter Object to store the output video in the disk.
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))
 
    # Declare a queue to store video frames.
    frames_queue = deque(maxlen = SEQUENCE_LENGTH)
    print("prediting start")
 
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
            predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis = 0))[0]
            print(f'predicted label probabil is {predicted_labels_probabilities}')
 
            # Get the index of class with highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)
            print(f'predicted label is {predicted_label}')
 
            # Get the class name using the retrieved index.
            predicted_class_name = CLASSES_LIST[predicted_label]
            print(f'predicted label is {predicted_class_name}')
 
        # Write predicted class name on top of the frame.
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
        # Write The frame into the disk using the VideoWriter Object.
        video_writer.write(frame)
        
    # Release the VideoCapture and VideoWriter objects.
    video_reader.release()
    print("prediting finished")
    video_writer.release()
    return output_file_path



# def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH):
    
    
#     # Initialize the VideoCapture object to read from the video file.
#     video_reader = cv2.VideoCapture(video_file_path)

#     # Get the width and height of the video.
#     original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
#     original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     # Initialize the VideoWriter Object to store the output video in the disk.
#     video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 
#                                    video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))
 
#     # Declare a queue to store video frames.
#     frames_queue = deque(maxlen = SEQUENCE_LENGTH)
#     print("prediting start")
 
#     # Initialize a variable to store the predicted action being performed in the video.
#     predicted_class_name = ''

#     # Iterate until the video is accessed successfully.
#     while video_reader.isOpened():
        
#         # Read the frame.
#         ok, frame = video_reader.read() 
        
#         # Check if frame is not read properly then break the loop.
#         if not ok:
#             break
 
#         # Resize the Frame to fixed Dimensions.
#         resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        
#         # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
#         normalized_frame = resized_frame / 255
 
#         # Appending the pre-processed frame into the frames list.
#         frames_queue.append(normalized_frame)
 
#         # Check if the number of frames in the queue are equal to the fixed sequence length.
#         if len(frames_queue) == SEQUENCE_LENGTH:
 
#             # Pass the normalized frames to the model and get the predicted probabilities.
#             predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis = 0))[0]
#             print(f'predicted label probabil is {predicted_labels_probabilities}')
 
#             # Get the index of class with highest probability.
#             predicted_label = np.argmax(predicted_labels_probabilities)
#             print(f'predicted label is {predicted_label}')
 
#             # Get the class name using the retrieved index.
#             predicted_class_name = CLASSES_LIST[predicted_label]
#             print(f'predicted label is {predicted_class_name}')
 
#         # Write predicted class name on top of the frame.
#         cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
#         # Write The frame into the disk using the VideoWriter Object.
#         video_writer.write(frame)
        
#     # Release the VideoCapture and VideoWriter objects.
#     video_reader.release()
#     print("prediting finished")
#     video_writer.release()
#     return output_file_path


def predict_single_action(video_file_path, SEQUENCE_LENGTH):
   
    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)
 
    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    # Declare a list to store video frames we will extract.
    frames_list = []
    
    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''
 
    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
 
    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)
 
    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):
 
        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
 
        # Read a frame.
        success, frame = video_reader.read() 
 
        # Check if frame is not read properly then break the loop.
        if not success:
            break
 
        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255
        
        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)
 
    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis = 0))[0]
    
    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)
 
    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]

          
    # Display the predicted action along with the prediction confidence.
    print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
        
    # Release the VideoCapture object. 
    video_reader.release()
    return predicted_class_name 



def send_msg_on_telegram(message,files):

    telegram_api_url=f"https://api.telegram.org/bot{telegram_auth_token}/sendPhoto?chat_id=@{telegram_group_id}&caption={message}"
    
    tel_res=requests.post(telegram_api_url,files=files)
    

    if tel_res.status_code==200:
        print("INFO: Message and Photo has been sent to the Telegram")
    else:
        print("ERROR: Could not send Message and Photo")



@app.route("/")
def hello_world():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect('/')
        file = request.files['file']

        if file.filename == '':
            flash('No image selected for uploading')
            return redirect('/')
        else:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print('upload_video filename: ' + filename)
            
            input_video_file_path=f'static/assets/upload_vedios/{filename}'
            

            # Construct the output video path.
            output_video_file_path = f'{test_videos_directory}/{filename.split(".")[0]}.avi'
            
            # Perform Action Recognition on the Test Video.
            complete=predict_on_video(input_video_file_path,output_video_file_path,SEQUENCE_LENGTH)
            
            filename=filename.split(".")[0]
            # Display the output video.
            # VideoFileClip(output_video_file_path, audio=False, target_resolution=(300,None)).ipython_display()

            return render_template('index.html', filename=filename,complete=complete)


@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    filename=filename+'.avi'
    print(app.root_path)
    full_path = os.path.join(app.root_path,'static/assets/test_vedios')
    print(full_path)
    return send_from_directory(full_path, filename)

            
@app.route('/live')
def live():
    # Open the camera
    camera = cv2.VideoCapture(0)

    fourcc=cv2.VideoWriter_fourcc(*'XVID')

    out=cv2.VideoWriter('Frame_0.mp4',fourcc,20.0,(640,480))

    # i variable is to give unique name to images
    i = 1
    wait = 0
    image_index=0
    
    while True:
        # Read video by read() function and it
        # will extract and  return the frame
        ret, img = camera.read()
    
        # Put current DateTime on each frame
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, str(datetime.now()), (20, 40),
                    font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(img)
    
        # Display the image
        cv2.imshow('live video', img)
    
        # wait for user to press any key
        key = cv2.waitKey(100)
    
        # wait variable is to calculate waiting time
        wait = wait+100

        if key == ord('q'):
            break
        # when it reaches to 5000 milliseconds
        # we will save that frame in given folder
        if wait == 10000:
            if (i-1)==0:
                predict_filename='Frame_0.mp4'
                print(predict_filename)   
            else:
                indexV=i-1
                predict_filename = 'Frame_'+str(indexV)+'.mp4'
                print(predict_filename)
            
            out.release() 

            # Perform Single Prediction on the Test Video.
            predicted_class_name = predict_single_action(predict_filename, SEQUENCE_LENGTH)

            

            if predicted_class_name=='Violence':
            
                vidcap = cv2.VideoCapture(predict_filename)
                # get total number of frames
                totalFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
                randomFrameNumber=random.randint(0, totalFrames)
                # set frame position
                vidcap.set(cv2.CAP_PROP_POS_FRAMES,randomFrameNumber)
                success, image = vidcap.read()
                if success:
                    print(f"-------------------------- {image_index}")
                    image_path=f'Violence_Images/random_frame-{image_index}.jpg'
                    cv2.imwrite(image_path, image)
                    image_index=image_index+1

                    msg = f"message recieved on {curr_date} at {curr_time}"
                    files={'photo':open(image_path,'rb')}
                    send_msg_on_telegram(msg,files)
                
                vidcap.release()
                
            else:
                print("not violence")

            filename = 'Frame_'+str(i)+'.mp4'

            fourcc=cv2.VideoWriter_fourcc(*'XVID')
            out=cv2.VideoWriter(filename,fourcc,20.0,(640,480))

            if os.path.isfile(predict_filename):
                os.remove(predict_filename)
            else:
                # If it fails, inform the user.
                print("Error: %s file not found" % predict_filename)
            
            i = i+1
            wait = 0
            
    # close the camera
    camera.release()
    out.release()
    # close open windows
    cv2.destroyAllWindows()   
    return redirect('/')

if __name__ == "__main__":
    app.run(debug=True)
