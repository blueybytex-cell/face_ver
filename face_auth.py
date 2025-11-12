import cv2
import numpy as np
import os
import pickle

class FaceAuthenticator:
    def __init__(self):
        self.models_loaded = False
        self.detector = None
        self.recognizer = None
        self.load_models()
    
    def load_models(self):
        """Load face detection and recognition models"""
        try:
            # Initialize face detector
            detector_path = "models/face_detection_yunet_2023mar.onnx"
            if not os.path.exists(detector_path):
                raise FileNotFoundError(f"Face detector model not found at {detector_path}")
            
            self.detector = cv2.FaceDetectorYN.create(
                detector_path,
                "",
                (320, 320),
                0.9,  # score threshold
                0.3,  # nms threshold
                5000  # top_k
            )
            
            # Initialize face recognizer
            recognizer_path = "models/face_recognition_sface_2021dec.onnx"
            if not os.path.exists(recognizer_path):
                raise FileNotFoundError(f"Face recognizer model not found at {recognizer_path}")
            
            self.recognizer = cv2.FaceRecognizerSF.create(
                recognizer_path,
                ""
            )
            
            self.models_loaded = True
            print("✅ Models loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.models_loaded = False
    
    def capture_face_embedding(self):
        """Capture face from camera and return embedding - WITH VISUAL FEEDBACK"""
        if not self.models_loaded:
            print("❌ Models not loaded")
            return None
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return None
        
        print("📷 Camera activated - Looking for face...")
        print("💡 Press SPACE to capture, ESC to cancel")
        
        embedding = None
        captured_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for processing (320x320 for model)
            frame_resized = cv2.resize(frame, (320, 320))
            
            # Create display frame (larger for better visibility)
            display_frame = cv2.resize(frame, (640, 480))
            
            # Update detector input size
            self.detector.setInputSize((320, 320))
            
            # Detect faces on the resized frame
            faces = self.detector.detect(frame_resized)
            
            face_detected = False
            if faces[1] is not None:
                for face in faces[1]:
                    # Extract face coordinates and confidence
                    bbox = face[0:4].astype(np.int32)
                    confidence = face[-1]
                    
                    # Scale bounding box coordinates from 320x320 to 640x480 display
                    scale_x = 640 / 320
                    scale_y = 480 / 320
                    
                    bbox_display = [
                        int(bbox[0] * scale_x),
                        int(bbox[1] * scale_y),
                        int(bbox[2] * scale_x),
                        int(bbox[3] * scale_y)
                    ]
                    
                    # Draw bounding box on display frame
                    color = (0, 255, 0)  # Green
                    thickness = 2
                    cv2.rectangle(display_frame, 
                                (bbox_display[0], bbox_display[1]), 
                                (bbox_display[0] + bbox_display[2], bbox_display[1] + bbox_display[3]), 
                                color, thickness)
                    
                    # Draw confidence text
                    text = f'Face: {confidence:.2f}'
                    cv2.putText(display_frame, text,
                              (bbox_display[0], bbox_display[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Store the face embedding for capture
                    if not face_detected:
                        aligned_face = self.recognizer.alignCrop(frame_resized, face)
                        embedding = self.recognizer.feature(aligned_face)
                        captured_frame = frame_resized.copy()
                        face_detected = True
            
            # Add instructions to display
            cv2.putText(display_frame, "Press SPACE to capture, ESC to cancel", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, "Position your face in the green box", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Face Registration - SEE YOUR FACE HERE', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE to capture
                if embedding is not None:
                    print("✅ Face captured successfully!")
                    break
                else:
                    print("❌ No face detected - please position your face in frame")
            elif key == 27:  # ESC to cancel
                embedding = None
                print("❌ Face capture cancelled")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if embedding is not None:
            # Ensure consistent 1D array shape for storage
            return embedding.flatten()
        return None
    
    def verify_face(self):
        """Verify face against database - WITH VISUAL FEEDBACK"""
        if not self.models_loaded:
            print("❌ Models not loaded")
            return None
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return None
        
        print("📷 Camera activated - Verifying face...")
        print("💡 Press SPACE to verify, ESC to cancel")
        
        embedding = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for processing (320x320 for model)
            frame_resized = cv2.resize(frame, (320, 320))
            
            # Create display frame (larger for better visibility)
            display_frame = cv2.resize(frame, (640, 480))
            
            # Update detector input size
            self.detector.setInputSize((320, 320))
            
            # Detect faces on the resized frame
            faces = self.detector.detect(frame_resized)
            
            face_detected = False
            if faces[1] is not None:
                for face in faces[1]:
                    # Extract face coordinates and confidence
                    bbox = face[0:4].astype(np.int32)
                    confidence = face[-1]
                    
                    # Scale bounding box coordinates from 320x320 to 640x480 display
                    scale_x = 640 / 320
                    scale_y = 480 / 320
                    
                    bbox_display = [
                        int(bbox[0] * scale_x),
                        int(bbox[1] * scale_y),
                        int(bbox[2] * scale_x),
                        int(bbox[3] * scale_y)
                    ]
                    
                    # Draw bounding box on display frame
                    color = (0, 255, 0)  # Green
                    thickness = 2
                    cv2.rectangle(display_frame, 
                                (bbox_display[0], bbox_display[1]), 
                                (bbox_display[0] + bbox_display[2], bbox_display[1] + bbox_display[3]), 
                                color, thickness)
                    
                    # Draw confidence text
                    text = f'Face: {confidence:.2f}'
                    cv2.putText(display_frame, text,
                              (bbox_display[0], bbox_display[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Store the face embedding for verification
                    if not face_detected:
                        aligned_face = self.recognizer.alignCrop(frame_resized, face)
                        embedding = self.recognizer.feature(aligned_face)
                        face_detected = True
            
            # Add instructions to display
            cv2.putText(display_frame, "Press SPACE to verify, ESC to cancel", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, "Position your face in the green box", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Face Verification - SEE YOUR FACE HERE', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE to verify
                if embedding is not None:
                    print("✅ Face captured for verification!")
                    break
                else:
                    print("❌ No face detected - please position your face in frame")
            elif key == 27:  # ESC to cancel
                embedding = None
                print("❌ Face verification cancelled")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if embedding is not None:
            # Ensure consistent 1D array shape for comparison
            return embedding.flatten()
        return None

    def get_embedding_shape(self, embedding):
        """Debug method to check embedding shape"""
        if embedding is not None:
            return embedding.shape
        return "None"