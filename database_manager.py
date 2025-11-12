import sqlite3
import os
import pickle
from datetime import datetime
import numpy as np

class DatabaseManager:
    def __init__(self):
        self.db_path = "database/user_data.db"
        self.init_database()
    
    def init_database(self):
        """Initialize database and tables"""
        os.makedirs("database", exist_ok=True)
        os.makedirs("user_images", exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nickname TEXT UNIQUE NOT NULL,
                phone_number TEXT,
                face_embedding BLOB NOT NULL,
                password_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create user_images table to track uploaded images
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                image_filename TEXT NOT NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_user(self, nickname, face_embedding, phone_number="", password_hash=""):
        """Register a new user with face data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Ensure embedding is 1D array before storage
            if face_embedding.ndim > 1:
                face_embedding = face_embedding.flatten()
            
            # Convert numpy array to bytes for storage
            face_blob = pickle.dumps(face_embedding)
            
            cursor.execute('''
                INSERT INTO users (nickname, phone_number, face_embedding, password_hash)
                VALUES (?, ?, ?, ?)
            ''', (nickname, phone_number, face_blob, password_hash))
            
            user_id = cursor.lastrowid
            
            # Create user's image directory
            user_dir = f"user_images/{nickname}"
            os.makedirs(user_dir, exist_ok=True)
            
            conn.commit()
            print(f"✅ User '{nickname}' registered with embedding shape: {face_embedding.shape}")
            return user_id
        except sqlite3.IntegrityError:
            return None  # Nickname already exists
        finally:
            conn.close()
    
    def get_user_by_face(self, face_embedding, similarity_threshold=0.6):
        """Find user by face embedding with similarity check"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, nickname, face_embedding FROM users")
        users = cursor.fetchall()
        conn.close()
        
        best_match = None
        best_similarity = 0
        
        for user_id, nickname, stored_embedding_blob in users:
            stored_embedding = pickle.loads(stored_embedding_blob)
            
            # Ensure both embeddings are 1D arrays
            if stored_embedding.ndim > 1:
                stored_embedding = stored_embedding.flatten()
            if face_embedding.ndim > 1:
                face_embedding = face_embedding.flatten()
            
            # Calculate cosine similarity
            similarity = self.calculate_similarity(stored_embedding, face_embedding)
            
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match = (user_id, nickname, similarity)
        
        return best_match
    
    def calculate_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        try:
            # Ensure both are 1D arrays
            embedding1 = embedding1.flatten()
            embedding2 = embedding2.flatten()
            
            # Check if embeddings have the same shape
            if embedding1.shape != embedding2.shape:
                print(f"❌ Shape mismatch: {embedding1.shape} vs {embedding2.shape}")
                return 0.0
            
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            # Avoid division by zero
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            return similarity
            
        except Exception as e:
            print(f"❌ Error calculating similarity: {e}")
            return 0.0
    
    def save_user_image(self, user_id, image_filename):
        """Save image reference for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Save to database
        cursor.execute('''
            INSERT INTO user_images (user_id, image_filename)
            VALUES (?, ?)
        ''', (user_id, image_filename))
        
        conn.commit()
        conn.close()
    
    def get_user_images(self, user_id):
        """Get all images for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT image_filename FROM user_images 
            WHERE user_id = ? ORDER BY uploaded_at DESC
        ''', (user_id,))
        
        images = [row[0] for row in cursor.fetchall()]
        conn.close()
        return images