import cv2
import face_recognition
import numpy as np
import pandas as pd
import qrcode
from datetime import datetime
import random
import os

class AdminBackend:
    def __init__(self):
        self.excel_path = "D:/projects @/smart patient medicine tracker/patient_data.xlsx"
        os.makedirs(os.path.dirname(self.excel_path), exist_ok=True)
        
    def process_face_image(self, image):
        """Process face image and return face encodings"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        # Draw rectangle around face
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            
        return image, face_encodings[0] if face_encodings else None

    def generate_patient_id(self):
        """Generate unique patient ID"""
        return f"PAT{random.randint(1000, 9999)}"

    def save_patient_data(self, name, age, phone, medicines, face_encoding):
        """Save patient data to Excel"""
        patient_id = self.generate_patient_id()
        
        data = {
            'Patient ID': [patient_id],
            'Name': [name],
            'Age': [age],
            'Phone': [phone],
            'Medicines': [', '.join(medicines)],
            'Face Encoding': [face_encoding.tolist() if face_encoding is not None else None],
            'Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }
        
        df = pd.DataFrame(data)
        
        if os.path.exists(self.excel_path):
            existing_df = pd.read_excel(self.excel_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_excel(self.excel_path, index=False)
        return patient_id

    def generate_qr_code(self, patient_id, name, age, phone, medicines):
        """Generate QR code for patient"""
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        data = f"ID: {patient_id}\nName: {name}\nAge: {age}\nPhone: {phone}\nMedicines: {', '.join(medicines)}"
        qr.add_data(data)
        qr.make(fit=True)
        qr_image = qr.make_image(fill_color="black", back_color="white")
        return qr_image
