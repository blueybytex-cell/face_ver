import cv2
import numpy as np
import os
import sys

class FaceScannerTest:
    def __init__(self):
        self.models_loaded = False
        self.detector = None
        self.load_models()
    
    def load_models(self):
        """Load face detection model"""
        try:
            detector_path = "models/face_detection_yunet_2023mar.onnx"
            if not os.path.exists(detector_path):
                print("❌ ERROR: Face detector model not found!")
                print(f"Please make sure this file exists: {detector_path}")
                return False
            
            # Initialize with correct input size
            self.detector = cv2.FaceDetectorYN.create(
                detector_path,
                "",
                (320, 320),  # This is the expected input size
                0.9,   # score threshold
                0.3,   # nms threshold
                5000   # top_k
            )
            self.models_loaded = True
            print("✅ Face detection model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def run_face_scanner(self):
        """Run real-time face scanner with camera feed"""
        if not self.models_loaded:
            print("❌ Models not loaded. Cannot run scanner.")
            return
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ ERROR: Cannot access camera!")
            print("Make sure your camera is connected and not being used by another application.")
            return
        
        print("\n" + "="*50)
        print("🎥 REAL-TIME FACE SCANNER STARTED")
        print("="*50)
        print("✅ Camera activated successfully!")
        print("👀 Looking for faces...")
        print("💡 Press 'S' to save current frame")
        print("💡 Press 'Q' or 'ESC' to quit")
        print("="*50)
        
        face_count = 0
        saved_images = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            # Resize frame to match model's expected input size (320x320)
            frame_resized = cv2.resize(frame, (320, 320))
            
            # Create a larger display frame for better visibility
            display_frame = cv2.resize(frame, (640, 480))
            
            # Update detector input size to match our resized frame
            self.detector.setInputSize((320, 320))
            
            # Detect faces on the 320x320 frame
            faces = self.detector.detect(frame_resized)
            
            face_count_current = 0
            if faces[1] is not None:
                face_count_current = len(faces[1])
                
                for i, face in enumerate(faces[1]):
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
                    text = f'Face {i+1}: {confidence:.3f}'
                    cv2.putText(display_frame, text,
                              (bbox_display[0], bbox_display[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw facial landmarks (if available)
                    if len(face) > 5:  # YuNet provides landmarks
                        landmarks = face[4:14].astype(np.int32).reshape((5, 2))
                        for (x, y) in landmarks:
                            # Scale landmarks to display size
                            x_display = int(x * scale_x)
                            y_display = int(y * scale_y)
                            cv2.circle(display_frame, (x_display, y_display), 3, (0, 0, 255), -1)
            
            # Update face count if changed
            if face_count != face_count_current:
                face_count = face_count_current
                if face_count > 0:
                    print(f"✅ Detected {face_count} face(s) in frame")
                else:
                    print("❌ No faces detected")
            
            # Add status information to display
            status_text = [
                f"Faces detected: {face_count}",
                "Press 'S' to save image",
                "Press 'Q' to quit"
            ]
            
            for i, text in enumerate(status_text):
                cv2.putText(display_frame, text,
                          (10, 30 + i * 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, text,
                          (10, 30 + i * 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            # Show frame
            cv2.imshow('Real-Time Face Scanner - SEE YOUR FACE HERE', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                print("👋 Scanner stopped by user")
                break
            elif key == ord('s'):  # Save current frame
                saved_images += 1
                filename = f"face_capture_{saved_images}.jpg"
                cv2.imwrite(filename, display_frame)  # Save the display frame (640x480)
                print(f"💾 Saved image as: {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n🎉 Scanner session ended. Saved {saved_images} images.")
        
        if saved_images > 0:
            print("📸 Check your project folder for the saved face images!")

def check_dependencies():
    """Check if all required packages are installed"""
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not installed. Run: pip install opencv-python")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
    except ImportError:
        print("❌ NumPy not installed. Run: pip install numpy")
        return False
    
    return True

def check_camera_access():
    """Check if camera is accessible"""
    print("🔍 Checking camera access...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            print("✅ Camera is working and accessible!")
            return True
        else:
            print("❌ Camera opened but cannot read frames")
            return False
    else:
        print("❌ Cannot access camera")
        print("Possible solutions:")
        print("1. Make sure no other app is using the camera")
        print("2. Check camera permissions")
        print("3. Try a different camera (if available)")
        return False

def main():
    print("🔧 Initializing Face Scanner Test...")
    
    # Check dependencies first
    if not check_dependencies():
        return
    
    # Then check camera
    if not check_camera_access():
        return
    
    # Initialize scanner
    scanner = FaceScannerTest()
    
    if scanner.models_loaded:
        scanner.run_face_scanner()
    else:
        print("❌ Failed to initialize face scanner")
        print("\n📋 TROUBLESHOOTING:")
        print("1. Make sure model files are in 'models/' folder:")
        print("   - face_detection_yunet_2023mar.onnx")
        print("   - face_recognition_sface_2021dec.onnx")
        print("2. Download from provided links if missing")
        print("3. Check file permissions")

if __name__ == "__main__":
    main()