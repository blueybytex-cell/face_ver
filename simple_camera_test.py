import cv2

print("Testing basic camera access...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
else:
    print("✅ Camera opened successfully!")
    
    ret, frame = cap.read()
    if ret:
        print("✅ Can read frames from camera")
        print(f"Frame size: {frame.shape}")
        cv2.imshow('Camera Test - Press any key', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ Cannot read frames")
    
    cap.release()

print("Test completed!")