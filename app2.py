import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

# Load the pre-trained emotion detection model
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model('model.h5')

# Define the emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Define the Streamlit app
st.title('Emotion Detection')

# Define the user interface
st.write('Please turn on the camera and face the camera and make sure you are visible. Your data is protected with us, you are not being recorded. Feel free and be assured.')

# Add a "Turn on camera" button
if st.button('Turn on camera'):
    # Access the user's camera
    cap = cv2.VideoCapture(0)
    # Define the video processing loop
    while cap.isOpened():
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame
        frame = cv2.resize(frame, (640, 480))
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        
        # Iterate over the detected faces
        for (x, y, w, h) in faces:
            # Extract the face ROI
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Resize the ROI to match the input shape of the model
            roi_gray = cv2.resize(roi_gray, (48, 48))
            
            # Reshape the image to match the input shape of the model
            input_image = np.reshape(roi_gray, (1, 48, 48, 1))
            
            # Normalize the image pixel values
            input_image = input_image / 255.0
            
            # Make a prediction using the pre-trained model
            prediction = model.predict(input_image)
            
            # Get the index of the highest prediction value
            max_index = np.argmax(prediction)
            
            # Get the corresponding emotion label
            emotion = emotion_labels[max_index]
            
            # Draw a bounding box around the detected face and label the emotion
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Display the frame with the bounding boxes and emotion labels
        st.image(frame, channels='BGR')
        
    # Release the camera
    cap.release()
# Add a "Close camera" button
if st.button('Close camera'):
    st.write('Thank you for using our services!')
    cv2.destroyAllWindows()


