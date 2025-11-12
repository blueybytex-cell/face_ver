import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import os
import shutil
from face_auth import FaceAuthenticator
from database_manager import DatabaseManager

class FacialAuthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Authentication App")
        self.root.geometry("800x600")
        
        # Initialize components
        self.face_auth = FaceAuthenticator()
        self.db_manager = DatabaseManager()
        self.current_user = None
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(2, weight=1)
        
        self.show_login_screen()
    
    def show_login_screen(self):
        """Show login/registration screen"""
        self.clear_screen()
        
        # Title
        title_label = ttk.Label(self.main_frame, text="Facial Authentication System", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=20)
        
        # Login section
        login_frame = ttk.LabelFrame(self.main_frame, text="Login with Face", padding="10")
        login_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        login_frame.columnconfigure(0, weight=1)
        
        login_btn = ttk.Button(login_frame, text="Scan Face to Login", 
                              command=self.login_with_face)
        login_btn.grid(row=0, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Registration section
        reg_frame = ttk.LabelFrame(self.main_frame, text="Register New User", padding="10")
        reg_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(reg_frame, text="Nickname:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.nickname_entry = ttk.Entry(reg_frame, width=30)
        self.nickname_entry.grid(row=0, column=1, pady=5, padx=5, sticky=(tk.W, tk.E))
        
        reg_btn = ttk.Button(reg_frame, text="Register with Face Scan", 
                            command=self.register_user)
        reg_btn.grid(row=1, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        reg_frame.columnconfigure(1, weight=1)
    
    def clear_screen(self):
        """Clear all widgets from main frame"""
        for widget in self.main_frame.winfo_children():
            widget.destroy()
    
    def register_user(self):
        """Register a new user with face authentication"""
        nickname = self.nickname_entry.get().strip()
    
        if not nickname:
            messagebox.showerror("Error", "Please enter a nickname")
            return
    
        print("🔄 Starting face registration...")
        print("📷 A camera window should open shortly...")
    
        # Capture face embedding
        embedding = self.face_auth.capture_face_embedding()
    
        if embedding is None:
             messagebox.showerror("Error", "Face capture failed or was cancelled")
             return
    
    # Rest of registration code...
        
        if not nickname:
            messagebox.showerror("Error", "Please enter a nickname")
            return
        
        print("🔄 Starting face registration...")
        
        # Capture face embedding
        embedding = self.face_auth.capture_face_embedding()
        
        if embedding is None:
            messagebox.showerror("Error", "Face capture failed or was cancelled")
            return
        
        # Register user in database
        user_id = self.db_manager.register_user(nickname, embedding)
        
        if user_id is None:
            messagebox.showerror("Error", "Nickname already exists")
            return
        
        messagebox.showinfo("Success", f"User '{nickname}' registered successfully!")
        self.current_user = {'id': user_id, 'nickname': nickname}
        self.show_main_app()
    
    def login_with_face(self):
        """Login user with facial recognition"""
        print("🔄 Starting face verification...")
        
        # Capture face for verification
        embedding = self.face_auth.verify_face()
        
        if embedding is None:
            messagebox.showerror("Error", "Face verification failed or was cancelled")
            return
        
        # Find matching user
        user_match = self.db_manager.get_user_by_face(embedding)
        
        if user_match:
            user_id, nickname, similarity = user_match
            self.current_user = {'id': user_id, 'nickname': nickname}
            messagebox.showinfo("Success", f"Welcome back {nickname}! (Similarity: {similarity:.2f})")
            self.show_main_app()
        else:
            messagebox.showerror("Error", "No matching user found")
    
    def show_main_app(self):
        """Show main application after login"""
        self.clear_screen()
        
        # Header with user info
        header_frame = ttk.Frame(self.main_frame)
        header_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(header_frame, text=f"Welcome, {self.current_user['nickname']}!", 
                 font=('Arial', 14, 'bold')).grid(row=0, column=0, sticky=tk.W)
        
        logout_btn = ttk.Button(header_frame, text="Logout", 
                               command=self.logout)
        logout_btn.grid(row=0, column=1, sticky=tk.E)
        
        header_frame.columnconfigure(0, weight=1)
        
        # Image management section
        img_frame = ttk.LabelFrame(self.main_frame, text="Image Storage", padding="10")
        img_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Upload button
        upload_btn = ttk.Button(img_frame, text="Upload Image", 
                               command=self.upload_image)
        upload_btn.grid(row=0, column=0, pady=10, sticky=tk.W)
        
        # Refresh button
        refresh_btn = ttk.Button(img_frame, text="Refresh Gallery", 
                                command=self.refresh_gallery)
        refresh_btn.grid(row=0, column=1, pady=10, padx=5, sticky=tk.W)
        
        # Gallery frame
        gallery_frame = ttk.Frame(img_frame)
        gallery_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Create scrollable canvas for gallery
        self.canvas = tk.Canvas(gallery_frame, bg='white')
        scrollbar = ttk.Scrollbar(gallery_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Configure grid weights
        self.main_frame.rowconfigure(1, weight=1)
        img_frame.rowconfigure(1, weight=1)
        img_frame.columnconfigure(0, weight=1)
        
        # Load initial gallery
        self.refresh_gallery()
    
    def upload_image(self):
        """Upload image to user's storage"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
           filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
         )
    
        if file_path:
            try: 
                # Create user directory if not exists
                user_dir = f"user_images/{self.current_user['nickname']}"
                os.makedirs(user_dir, exist_ok=True)
            
                # Generate unique filename
                filename = os.path.basename(file_path)
                counter = 1
                name, ext = os.path.splitext(filename)
            
                while os.path.exists(os.path.join(user_dir, filename)):
                    filename = f"{name}_{counter}{ext}"
                    counter += 1
            
                 # Copy file to user directory
                dest_path = os.path.join(user_dir, filename)
                shutil.copy2(file_path, dest_path)
            
                # Save to database - FIXED: Now passing correct arguments
                self.db_manager.save_user_image(self.current_user['id'], filename)
                
                messagebox.showinfo("Success", "Image uploaded successfully!")
                self.refresh_gallery()
            
            except Exception as e:
                 messagebox.showerror("Error", f"Failed to upload image: {e}")

    def refresh_gallery(self):
        """Refresh the image gallery"""
        # Clear existing gallery
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Get user's images
        images = self.db_manager.get_user_images(self.current_user['id'])
        user_dir = f"user_images/{self.current_user['nickname']}"
        
        if not images:
            no_images_label = ttk.Label(self.scrollable_frame, text="No images uploaded yet")
            no_images_label.grid(row=0, column=0, pady=20)
            return
        
        # Display images in gallery
        row, col = 0, 0
        max_cols = 3
        
        for idx, image_filename in enumerate(images):
            image_path = os.path.join(user_dir, image_filename)
            
            if os.path.exists(image_path):
                try:
                    # Load and resize image for thumbnail
                    img = Image.open(image_path)
                    img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    
                    # Create image frame
                    img_frame = ttk.Frame(self.scrollable_frame, relief="solid", borderwidth=1)
                    img_frame.grid(row=row, column=col, padx=5, pady=5, sticky=(tk.W, tk.E))
                    
                    # Image label
                    img_label = ttk.Label(img_frame, image=photo)
                    img_label.image = photo  # Keep reference
                    img_label.grid(row=0, column=0, padx=5, pady=5)
                    
                    # Filename label
                    name_label = ttk.Label(img_frame, text=image_filename[:20] + "..." if len(image_filename) > 20 else image_filename)
                    name_label.grid(row=1, column=0, padx=5, pady=2)
                    
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
                        
                except Exception as e:
                    print(f"Error loading image {image_filename}: {e}")
    
    def logout(self):
        """Logout current user"""
        self.current_user = None
        self.show_login_screen()

def main():
    # Check if models exist
    if not os.path.exists("models/face_detection_yunet_2023mar.onnx"):
        print("❌ ERROR: Model files not found!")
        print("Please download these files and place in 'models/' folder:")
        print("1. face_detection_yunet_2023mar.onnx")
        print("2. face_recognition_sface_2021dec.onnx")
        print("\nDownload from: https://github.com/opencv/opencv_zoo/tree/master/models")
        return
    
    root = tk.Tk()
    app = FacialAuthApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()