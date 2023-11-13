import cv2 as cv 
import numpy as np3
import matplotlib.pyplot as plt3
from tensorflow import keras
from keras import datasets, layers , models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import shutil



(training_images , training_labels) , (testing_images , testing_labels) = datasets.cifar10.load_data()

#Normalizing the images
training_images , testing_images = training_images / 255 , testing_images / 255

#Splitting the data into training data and testing data
#Path of original dataset
basedir = r'C:\\Users\\91998\\Downloads\\CS370_HOMEWORK3_FOLDER\\drone_dataset_yolo\\dataset_txt'
traindir = r'C:\Users\91998\Downloads\CS370_HOMEWORK3_FOLDER\output_dataset\train'
traindronesdir = os.path.join(traindir , 'drones_train')
testdir = r'C:\Users\91998\Downloads\CS370_HOMEWORK3_FOLDER\output_dataset\test'
testdronesdir = os.path.join(testdir , 'drones_test')


test_size = 0.2

os.makedirs(traindronesdir , exist_ok = True)
os.makedirs(testdronesdir , exist_ok = True)

all_images = [img for img in os.listdir(basedir) if img.endswith('.jpg')]

# Split the images into training and testing sets
train_images, test_images = train_test_split(all_images, test_size=test_size, random_state=42)

#For loop for training images:
for image_name in train_images:
    input_image_path = os.path.join(basedir , image_name)
    output_image_path = os.path.join(traindronesdir , image_name)
    shutil.copy(input_image_path , output_image_path)

for img_name in test_images:
    input_img_path = os.path.join(basedir , img_name)
    output_img_path = os.path.join(testdronesdir , img_name)
    shutil.copy(input_img_path , output_img_path)
    
#The dataset has been split into training and testing images.

#Training the CNN model
train_data_dir = r"C:\\Users\\91998\\Downloads\\CS370_HOMEWORK3_FOLDER\\output_dataset\\train"
test_data_dir = r"C:\\Users\\91998\\Downloads\\CS370_HOMEWORK3_FOLDER\\output_dataset\\test"

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(learning_rate=1e-4) , loss='binary_crossentropy', metrics=['acc'])

#Image augmentation
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_datagenerator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (150 , 150),
    batch_size = 20,
    class_mode='binary')

test_datagenerator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(150 , 150),
    batch_size=20,
    class_mode='binary')


print(os.listdir(train_data_dir))
print(os.listdir(test_data_dir))


epochs = 10
model.fit(train_datagenerator, epochs=epochs)

'''loss, accuracy = model.evaluate(test_datagenerator)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')'''

model.save('drone_detection_model.h5')


import cv2
import os
import numpy as np3
from keras.models import load_model

def detect_drones(frame , model):
    frame = cv2.resize(frame , (150 , 150))
    frame = frame / 255.0
    
    
    input_frame = np3.expand_dims(frame , axis = 0)
    predictions = model.predict(input_frame)
    
    confidence_score = predictions[0]
    
    threshold = 0.00000000414                            #Adjusted as observed
    if confidence_score > threshold:
        return True
    else:
        return False


def process_video(video_path , output_directory , model):
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while cap.isOpened():
        ret , frame = cap.read()
        if not ret:
            break
        
        is_drone_detected = detect_drones(frame, model)
        
        if is_drone_detected:
            save_path = os.path.join(output_directory, f"detections_{frame_count}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"Saved frame {frame_count}")

            
        frame_count += 1
        
    cap.release()
    
def process_videos(input_directory, output_directory, model):
    for filename in os.listdir(input_directory):
        if filename.endswith(".mp4"):
            video_path = os.path.join(input_directory, filename)
            process_video(video_path, output_directory, model)
    
video_path = "C:/Users/91998/Downloads/CS370_HOMEWORK3_FOLDER"
output_directory = "C:/Users/91998/Downloads/CS370_HOMEWORK3_FOLDER/detections"
model_path = "C:/Users/91998/Downloads/CS370_HOMEWORK3_FOLDER/drone_detection_model.h5"
trained_model = load_model(model_path)

process_videos(video_path , output_directory , trained_model)


















