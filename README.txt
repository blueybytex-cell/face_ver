FACIAL AUTHENTICATION APP - SETUP INSTRUCTIONS

1. DOWNLOAD REQUIRED MODEL FILES:
   - Go to: https://github.com/opencv/opencv_zoo/tree/master/models
   - Download these two files:
     * face_detection_yunet_2023mar.onnx
     * face_recognition_sface_2021dec.onnx

2. CREATE FOLDER STRUCTURE:
   - Create project folder: facial_auth_app/
   - Inside it, create: models/
   - Place the downloaded .onnx files in models/

3. INSTALL DEPENDENCIES:
   pip install opencv-python pillow numpy

4. RUN THE APPLICATION:
   python main.py

FEATURES:
- Register new users with face scan
- Login with facial recognition  
- Personal image storage for each user
- Shared gallery - all users can see all images
- Local SQLite database (no internet required)
- Complete GUI interface

SECURITY:
- Face embeddings stored locally
- Each user has personal storage space
- Automatic face matching with similarity scoring