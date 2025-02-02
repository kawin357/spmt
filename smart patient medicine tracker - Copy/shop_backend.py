import cv2
import face_recognition
import numpy as np
import pandas as pd
import qrcode
from PIL import Image
import os

class ShopBackend:
    def __init__(self):
        self.excel_path = "D:/projects @/smart patient medicine tracker/patient_data.xlsx"
        self.payment_qr_path = "D:/projects @/smart patient medicine tracker/payment_qr.jpg"
        
    def verify_face(self, image):
        """Verify face against stored faces"""
        if not os.path.exists(self.excel_path):
            return None, "No patient records found"
            
        # Load patient data
        df = pd.read_excel(self.excel_path)
        
        # Get face encoding of the current image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            return None, "No face detected in image"
            
        face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
        
        # Draw rectangle around detected face
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Compare with stored faces
        for idx, row in df.iterrows():
            stored_encoding = np.array(eval(str(row['Face Encoding']))) if row['Face Encoding'] else None
            if stored_encoding is not None:
                match = face_recognition.compare_faces([stored_encoding], face_encoding, tolerance=0.6)[0]
                if match:
                    return row.to_dict(), image
                    
        return None, image

    def verify_qr(self, qr_image):
        """Verify QR code and return patient data"""
        # This is a placeholder. In a real application, you'd need to:
        # 1. Decode the QR code
        # 2. Extract the patient ID
        # 3. Look up the patient data in the Excel file
        return None

    def generate_bill(self, medicines, days):
        """Generate bill and payment QR code"""
        # Simple random price generation
        price_per_med = {med: random.randint(10, 100) for med in medicines}
        total = sum(price_per_med.values()) * days
        
        # Generate payment QR
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(f"Amount: ₹{total}")
        qr.make(fit=True)
        qr_image = qr.make_image(fill_color="black", back_color="white")
        qr_image.save(self.payment_qr_path)
        
        return {
            'prices': price_per_med,
            'days': days,
            'total': total,
            'qr_path': self.payment_qr_path
        }
