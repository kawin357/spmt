import streamlit as st
st.set_page_config(page_title="Smart Patient Medicine Tracker", layout="wide")

import cv2
import face_recognition
import numpy as np
import pandas as pd
import qrcode
from PIL import Image
import os
import random
import time
import io
from datetime import datetime
import json
import base64
from pyzbar import pyzbar
import matplotlib.pyplot as plt
import seaborn as sns
import decode
from datetime import datetime

# Custom CSS for animations and styling
st.markdown("""
<style>
      .sidebar .sidebar-content {
    background-image: linear-gradient(#4CAF50, #81C784);
    color: white;
}

.big-font {
    font-size: 35px !important;
    font-weight: bold;
    animation: fadeIn 1.5s;
    color: #ffffff;
    text-align: center;
    margin-bottom: 20px;
}

.menu-item {
    
  font-size: 24px !important;
    font-weight: bold;
    padding: 15px 20px;
    margin: 10px 0;
    border-radius: 10px;
    transition: all 0.3s;
    background-color: rgba(195, 190, 190, 0.329);
    border: 2px solid #4CAF50;
    cursor: pointer;
    color: #4CAF50;
    text-align: center;
    text-decoration: none;
    display: block;
}

.menu-item:hover {
    background-color: #4CAF50;
    color: white;
    transform: scale(1.02);
    box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
}

.menu-item.active {
    background-color: #4CAF50;
    color: white;
    box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
}

.menu-item:hover {
    background-color: rgba(255,255,255,0.1);
    transform: scale(1.02);
}

.page-title {
    font-size: 40px !important;
    font-weight: bold;
    color: #4CAF50;
    text-align: center;
    margin: 20px 0;
    animation: slideInDown 1s;
}

.section-title {
    font-size: 28px !important;
    color: #4CAF50;
    margin: 15px 0;
    animation: fadeIn 1s;
}

.instruction-box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    margin: 15px 0;
    animation: slideIn 1s;
    color: #000000;
    box-shadow: 0 2px 4px rgba(76, 175, 80, 0.2);
    transition: transform 0.3s;
}

.instruction-box:hover {
    transform: translateY(-5px);
    box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
}

.process-step {
    padding: 10px;
    margin: 5px 0;
    border-left: 3px solid #4CAF50;
    background-color: rgba(76, 175, 80, 0.1);
}

.stTable {
    background-color: white !important;
}

.stTable th {
    background-color: #4CAF50 !important;
    color:  #063a09 !important;
}

.stTable td {
    color:  #063a09 !important;
}

.button-primary {
    background-color: #4CAF50;
    color: white;
    padding: 12px 24px;
    border-radius: 5px;
    border: none;
    font-size: 18px;
    cursor: pointer;
    transition: all 0.3s;
}

.button-primary:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
}

.stButton>button {
    background-color: #4CAF50 !important;
    color: white !important;
    font-size: 18px !important;
    padding: 12px 24px !important;
    border-radius: 5px !important;
    transition: all 0.3s !important;
}

.stButton>button:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3) !important;
}

@keyframes fadeIn {
    0% { opacity: 0; }
    100% { opacity: 1; }
}

@keyframes slideIn {
    0% { transform: translateX(-100%); opacity: 0; }
    100% { transform: translateX(0); opacity: 1; }
}

@keyframes slideInDown {
    0% { transform: translateY(-50px); opacity: 0; }
    100% { transform: translateY(0); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

def show_process_step(title, description):
    st.markdown(f"""
    <div class="process-step">
        <strong>{title}</strong><br>
        {description}
    </div>
    """, unsafe_allow_html=True)

def generate_qr_code(data, filename):
    try:
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(data)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_img.save(filename)
        return True
    except Exception as e:
        st.error(f"Error generating QR code: {str(e)}")
        return False

def show_face_encoding_process(img_array):
    with st.spinner("Processing face..."):
        col1, col2 = st.columns(2)
        with col1:
            show_process_step("Step 1: Image Loading", "Converting image to numerical array")
            time.sleep(0.5)
            
            show_process_step("Step 2: Face Detection", "Locating face in the image")
            face_locations = face_recognition.face_locations(img_array)
            time.sleep(0.5)
            
            show_process_step("Step 3: Feature Extraction", "Identifying facial landmarks")
            time.sleep(0.5)
            
            show_process_step("Step 4: Encoding Generation", "Converting to 128-dimensional vector")
            face_encodings = face_recognition.face_encodings(img_array, face_locations)
            time.sleep(0.5)
            
            st.success("Face encoding completed!")
            return face_locations, face_encodings

def download_button(data, filename, button_text):
    # Convert DataFrame to CSV
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}"> {button_text}</a>'
    st.markdown(href, unsafe_allow_html=True)

def show_horizontal_process(steps):
    cols = st.columns(len(steps))
    for idx, (col, (title, desc)) in enumerate(zip(cols, steps.items())):
        with col:
            st.markdown(f"""
            <div class="process-step" style="text-align: center;">
                <div style="font-size: 24px; margin-bottom: 10px;">⚪</div>
                <strong>{title}</strong><br>
                <small>{desc}</small>
                <div class="process-line"></div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.3)
            st.markdown(f"""
            <div class="process-step" style="text-align: center;">
                <div style="font-size: 24px; margin-bottom: 10px; color: #2e7bcf;">⚫</div>
            </div>
            """, unsafe_allow_html=True)

def save_table_as_image(df, filepath):
    # Create figure and axis with larger size
    plt.figure(figsize=(12, len(df)*0.5+2))
    
    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Create table
    table = plt.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='left',
        loc='center',
        colColours=['#2e7bcf']*len(df.columns),
        cellColours=[['white']*len(df.columns)]*len(df)
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color text in header white
    for key, cell in table._cells.items():
        if key[0] == 0:  # Header row
            cell.set_text_props(color='white')
    
    # Save figure
    plt.savefig(filepath, bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.close()

def format_bill_data(bill_data):
    # Convert all values to strings to avoid type conversion issues
    formatted_data = {
        "Field": [],
        "Value": []
    }
    
    for key, value in bill_data.items():
        formatted_data["Field"].append(key)
        formatted_data["Value"].append(str(value[0]))
    
    return pd.DataFrame(formatted_data)

def main():
    # Create directories if they don't exist
    os.makedirs("D:/projects @/smart patient medicine tracker", exist_ok=True)
    os.makedirs("D:/projects @/smart patient medicine tracker/patient_details", exist_ok=True)
    os.makedirs("D:/projects @/smart patient medicine tracker/patient_qrcodes", exist_ok=True)
    os.makedirs("D:/projects @/smart patient medicine tracker/patient_bills", exist_ok=True)
    
    # Initialize Excel file if it doesn't exist
    excel_path = "D:/projects @/smart patient medicine tracker/patient_data.xlsx"
    if not os.path.exists(excel_path):
        initial_df = pd.DataFrame(columns=[
            'S.No', 'Patient_ID', 'Name', 'Age', 'Gender', 'Phone', 'Medicines', 
            'Registration Date', 'Face_Encoding'
        ])
        initial_df.to_excel(excel_path, index=False)
    
    # Initialize session state for navigation if not exists
    if 'nav_page' not in st.session_state:
        st.session_state.nav_page = " Home"
    
    with st.sidebar:
        st.markdown('<p class="big-font">Navigation Menu</p>', unsafe_allow_html=True)
        
        # Create clickable navigation buttons using Streamlit buttons
        for option in [" Home", " Hospital side", " Medical Shop Side"]:
            # Use Streamlit button with custom styling
            if st.button(option, key=f"nav_{option}", use_container_width=True):
                st.session_state.nav_page = option
                st.rerun()
            
            # Add spacing between buttons
            st.markdown("<br>", unsafe_allow_html=True)
    
    # Navigate to the selected page
    if st.session_state.nav_page == " Home":
        show_home()
    elif st.session_state.nav_page == " Hospital side":
        admin_side()
    elif st.session_state.nav_page == " Medical Shop Side":
        medical_shop_side()

def show_home():
    st.markdown('<p class="big-font">Welcome to Smart Patient Medicine Tracker</p>', unsafe_allow_html=True)
    
    # Project Description
    st.markdown("""
    <div class="instruction-box">
        <h2>About This Project</h2>
        <p>Smart Patient Medicine Tracker is an innovative healthcare management system that uses facial recognition 
        technology to streamline patient identification and medicine dispensing processes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features
    st.markdown("""
    <div class="instruction-box">
        <h2>Key Features</h2>
        <ul>
            <li>Face Recognition Based Patient Identification</li>
            <li>Automatic Patient Details Retrieval</li>
            <li>Digital Prescription Management</li>
            <li>QR Code Based Payment System</li>
            <li>Secure Patient Data Storage</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # How to Use
    st.markdown("""
    <div class="instruction-box">
        <h2>How to Use</h2>
        <h3>For Hospital Admin:</h3>
        <ol>
            <li>Select "Admin Side" from the sidebar</li>
            <li>Enter patient details and capture photo</li>
            <li>Add prescribed medicines</li>
            <li>Submit to register the patient</li>
        </ol>
        <h3>For Medical Shop:</h3>
        <ol>
            <li>Select "Medical Shop Side" from the sidebar</li>
            <li>Capture patient's face for identification</li>
            <li>View patient details and prescriptions</li>
            <li>Select medicines and generate bill</li>
            <li>Process payment using QR code</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Contact Information
    st.markdown("""
    <div class="instruction-box">
        <h2>Contact :</h2>
        <p>For technical support or queries, please contact:</p>
        <p>Name : Kawin M.S</p>
        <p> Email: mskawin@gmail.com</p>
        <p> Phone:+91 8015355914</p>
        <p>GitHub Link: <a href="https://github.com/kawin789" target="_blank">GitHub</a></p>
<p>LinkedIn Page Link: <a href="www.linkedin.com/in/kawin-m-s-570961285" target="_blank">LinkedIn</a></p>
<p>Portfolio Link: <a href="https://kawin-portfolio.netlify.app/" target="_blank">Portfolio</a></p></div>
    """, unsafe_allow_html=True)

def admin_side():
    
    st.markdown("""
    <div class="instruction-box" style="text-align: center;">
        <h2>Welcome to Hospital Portal</h2>
        <p>Enter the patient details and capture photo</p>
    </div>
     <style>
     .stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 22px;
        
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Patient Information Form
    name = st.text_input("Patient Name")
    
    # Check for duplicate name
    if name:
        try:
            df = pd.read_excel("D:/projects @/smart patient medicine tracker/patient_data.xlsx")
            if not df.empty and name in df['Name'].values:
                st.error(f"Patient with name '{name}' already exists!")
                return
        except Exception as e:
            st.error(f"Error checking name: {str(e)}")
    
    age = st.number_input("Age", min_value=0, max_value=120)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    phone = st.text_input("Phone Number")
    
    num_medicines = st.number_input("Number of Medicines", min_value=1, max_value=4, value=1)
    
    medicines = []
    for i in range(int(num_medicines)):
        med = st.text_input(f"Medicine {i+1} Name")
        medicines.append(med)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(" Capture or Upload Patient Photo")
        picture = st.camera_input("Take a picture") or st.file_uploader("Or Upload Photo", 
                                                                      type=['png', 'jpg', 'jpeg'])
        
        if picture:
            img = Image.open(picture)
            img_array = np.array(img)
            
            # Show face encoding process with animation
            face_locations, face_encodings = show_face_encoding_process(img_array)
            
            if len(face_locations) == 0:
                st.warning("No face detected. Please try again with a clearer photo.")
            else:
                preview_img = img_array.copy()
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(preview_img, (left, top), (right, bottom), (0, 255, 0), 2)
                    if name:
                        cv2.putText(preview_img, name, (left, top-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                st.image(preview_img, caption="Captured Image with Face Detection")
                if len(face_encodings) > 0:
                    st.session_state['face_encodings'] = face_encodings[0].tolist()
                    st.session_state['captured_image'] = img_array
    
    if st.button("Submit"):
        if 'face_encodings' not in st.session_state:
            st.error("Please capture a photo first!")
            return
            
        try:
            df = pd.read_excel("D:/projects @/smart patient medicine tracker/patient_data.xlsx")
            new_sno = len(df) + 1 if not df.empty else 1
            patient_id = f"PAT{new_sno:04d}"
            
            new_data = {
                'S.No': [new_sno],
                'Patient_ID': [patient_id],
                'Name': [name],
                'Age': [age],
                'Gender': [gender],
                'Phone': [phone],
                'Medicines': [', '.join(medicines)],
                'Registration Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                'Face_Encoding': [json.dumps(st.session_state['face_encodings'])]
            }
            
            new_df = pd.DataFrame(new_data)
            
            # Generate and save QR Code
            qr_data = f"Patient ID : {patient_id}\n Name: {name}\n Age: {age}\n Gender: {gender}\n Phone: {phone}"
            qr_path = f"D:/projects @/smart patient medicine tracker/patient_qrcodes/{patient_id}_qr.png"
            
            if generate_qr_code(qr_data, qr_path):
                st.success("Patient registered successfully!")
                st.write("### Patient Details")
                
                details_data = {
                    "Serial Number": new_sno,
                    "Patient ID": patient_id,
                    "Name": name,
                    "Age": age,
                    "Gender": gender,
                    "Phone": phone,
                    "Prescribed Medicines": ', '.join(medicines)
                }
                
                details_df = pd.DataFrame(details_data.items(), columns=['Field', 'Value'])
                st.table(details_df.set_index('Field'))
                
                # Save patient details as image
                details_img_path = f"D:/projects @/smart patient medicine tracker/patient_details/{patient_id}_details.png"
                save_table_as_image(details_df, details_img_path)
                
                # Download buttons
                st.download_button(
                    label=" Download Patient Details (CSV)",
                    data=details_df.to_csv(index=False),
                    file_name=f"{patient_id}_details.csv",
                    mime="text/csv"
                )
                
                with open(qr_path, "rb") as f:
                    st.download_button(
                        label=" Download QR Code",
                        data=f.read(),
                        file_name=f"{patient_id}_qr.png",
                        mime="image/png"
                    )
                
                st.write("### Patient QR Code")
                st.image(qr_path, caption=f"QR Code - {patient_id}", width=300)
                
                if not df.empty:
                    df = pd.concat([df, new_df], ignore_index=True)
                else:
                    df = new_df
                
                df = df[['S.No', 'Patient_ID', 'Name', 'Age', 'Gender', 'Phone', 'Medicines', 
                        'Registration Date', 'Face_Encoding']]
                df.to_excel("D:/projects @/smart patient medicine tracker/patient_data.xlsx", index=False)
            
        except Exception as e:
            st.error(f"Error saving data: {str(e)}")

def medical_shop_side():
    
    
    # Add background text and option buttons
    st.markdown("""
    <div class="instruction-box" style="text-align: center;">
        <h2>Welcome to Medical Shop Portal</h2>
        <p>Choose your preferred method of patient identification</p>
    </div>
     <style>
     .stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 22px;
        
    }
    </style>
    """, unsafe_allow_html=True)
    
    method = st.radio("Select Identification Method",
    ["Face Recognition", "QR Code Scanner"],
    format_func=lambda x: f"🔍 {x}" if "Face" in x else f"📷 {x}"
)

    if method == "Face Recognition":
        st.write("### Face Recognition")
        picture = st.camera_input("Take a picture of the patient")
        
        if picture:
            steps = {
                "Image Capture": "Taking photo",
                "Processing": "Converting to array",
                "Face Detection": "Finding faces",
                "Encoding": "Generating features",
                "Database Search": "Finding matches"
            }
            show_horizontal_process(steps)
            
            img = Image.open(picture)
            img_array = np.array(img)
            face_locations = face_recognition.face_locations(img_array)
            
            if len(face_locations) == 0:
                st.warning("No face detected in the image. Please try again.")
                return
                
            face_encodings = face_recognition.face_encodings(img_array)[0]
            
            # Search for matching patient
            df = pd.read_excel("D:/projects @/smart patient medicine tracker/patient_data.xlsx")
            found = False
            
            for idx, row in df.iterrows():
                stored_encoding = np.array(json.loads(row['Face_Encoding']))
                match = face_recognition.compare_faces([stored_encoding], face_encodings)[0]
                
                if match:
                    found = True
                    st.success(f"Patient Identified: {row['Name']}")
                    
                    # Display patient details
                    details_data = {
                        "Patient ID": row['Patient_ID'],
                        "Name": row['Name'],
                        "Age": row['Age'],
                        "Gender": row['Gender'],
                        "Phone": row['Phone'],
                        "Prescribed Medicines": row['Medicines']
                    }
                    details_df = pd.DataFrame(details_data.items(), columns=['Field', 'Value'])
                    st.table(details_df.set_index('Field'))
                    
                    # Download button for patient details
                    download_button(details_df, f"{row['Patient_ID']}_details.csv", 
                                 "Download Patient Details")
                    
                    # Medicine selection
                    st.write("### Medicine Selection")
                    prescribed_meds = row['Medicines'].split(', ')
                    selected_med = st.selectbox("Select Medicine", prescribed_meds)
                    
                    if selected_med:
                        days = st.number_input(f"Number of days for {selected_med}", 
                                            min_value=1, max_value=30, value=1)
                        
                        if st.button("Generate Bill"):
                            bill_number = f"BILL{random.randint(1000, 9999)}"
                            price_per_day = random.randint(10, 50)
                            total = price_per_day * days
                            
                            bill_data = {
                                "Bill Number": [bill_number],
                                "Patient ID": [row['Patient_ID']],
                                "Patient Name": [row['Name']],
                                "Medicine": [selected_med],
                                "Price per day": [f"₹{price_per_day}"],
                                "Number of days": [str(days)],
                                "Total Amount": [f"₹{total}"],
                                "Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                            }
                            bill_df = pd.DataFrame(bill_data)
                            
                            # Create patient bills directory if not exists
                            patient_bills_dir = f"D:/projects @/smart patient medicine tracker/patient_bills/{row['Patient_ID']}"
                            os.makedirs(patient_bills_dir, exist_ok=True)
                            
                            # Save bill with timestamp
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            bill_filename = f"{bill_number}_{timestamp}.csv"
                            bill_path = f"{patient_bills_dir}/{bill_filename}"
                            bill_df.to_csv(bill_path, index=False)
                            
                            # Display bill vertically with proper formatting
                            st.write("### Bill Details")
                            vertical_bill_df = format_bill_data(bill_data)
                            st.table(vertical_bill_df.set_index('Field'))
                            
                            # Save bill as image
                            bill_img_path = f"{patient_bills_dir}/{bill_number}_{timestamp}.png"
                            save_table_as_image(vertical_bill_df, bill_img_path)
                            
                            # Download button for bill
                            st.download_button(
                                label=" Download Bill",
                                data=bill_df.to_csv(index=False),
                                file_name=bill_filename,
                                mime="text/csv"
                            )

                            # Display payment QR
                            payment_qr_path = "D:/projects @/smart patient medicine tracker/paymt.qr.jpg"
                            if os.path.exists(payment_qr_path):
                                st.write("### Scan QR Code to Pay")
                                st.image(payment_qr_path, width=300)
                    break
            
            if not found:
                st.error("Patient not found in database")
    
    else:  # QR Code Scanner
        st.write("### QR Code Scanner")
        qr_image = st.camera_input("Scan QR Code") or st.file_uploader("Or Upload QR Code", 
                                                                      type=['png', 'jpg', 'jpeg'])
        
        if qr_image:
            steps = {
                "Image Capture": "Reading QR",
                "Decoding": "Extracting data",
                "Validation": "Checking format",
                "Database": "Finding patient",
                "Display": "Showing details"
            }
            show_horizontal_process(steps)
            
            try:
                img = Image.open(qr_image)
                decoded_objects = pyzbar.decode(img)
                
                if decoded_objects:
                    qr_data = decoded_objects[0].data.decode('utf-8')
                    patient_id = qr_data.split('\n')[0].split(': ')[1]
                    
                    df = pd.read_excel("D:/projects @/smart patient medicine tracker/patient_data.xlsx")
                    patient = df[df['Patient_ID'] == patient_id].iloc[0]
                    
                    st.success(f"Patient Identified: {patient['Name']}")
                    
                    # Rest of the code same as face recognition section
                    # (Display patient details, medicine selection, bill generation)
                    details_data = {
                        "Patient ID": patient['Patient_ID'],
                        "Name": patient['Name'],
                        "Age": patient['Age'],
                        "Gender": patient['Gender'],
                        "Phone": patient['Phone'],
                        "Prescribed Medicines": patient['Medicines']
                    }
                    details_df = pd.DataFrame(details_data.items(), columns=['Field', 'Value'])
                    st.table(details_df.set_index('Field'))
                    
                    # Download button for patient details
                    download_button(details_df, f"{patient['Patient_ID']}_details.csv", 
                                 "Download Patient Details")
                    
                    # Medicine selection
                    st.write("### Medicine Selection")
                    prescribed_meds = patient['Medicines'].split(', ')
                    selected_med = st.selectbox("Select Medicine", prescribed_meds)
                    
                    if selected_med:
                        days = st.number_input(f"Number of days for {selected_med}", 
                                            min_value=1, max_value=30, value=1)
                        
                        if st.button("Generate Bill"):
                            bill_number = f"BILL{random.randint(1000, 9999)}"
                            price_per_day = random.randint(10, 50)
                            total = price_per_day * days
                            
                            bill_data = {
                                "Bill Number": [bill_number],
                                "Patient ID": [patient['Patient_ID']],
                                "Patient Name": [patient['Name']],
                                "Medicine": [selected_med],
                                "Price per day": [f"₹{price_per_day}"],
                                "Number of days": [str(days)],
                                "Total Amount": [f"₹{total}"],
                                "Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                            }
                            bill_df = pd.DataFrame(bill_data)
                            
                            # Create patient bills directory if not exists
                            patient_bills_dir = f"D:/projects @/smart patient medicine tracker/patient_bills/{patient['Patient_ID']}"
                            os.makedirs(patient_bills_dir, exist_ok=True)
                            
                            # Save bill with timestamp
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            bill_filename = f"{bill_number}_{timestamp}.csv"
                            bill_path = f"{patient_bills_dir}/{bill_filename}"
                            bill_df.to_csv(bill_path, index=False)
                            
                            # Display bill vertically with proper formatting
                            st.write("### Bill Details")
                            vertical_bill_df = format_bill_data(bill_data)
                            st.table(vertical_bill_df.set_index('Field'))
                            
                            # Save bill as image
                            bill_img_path = f"{patient_bills_dir}/{bill_number}_{timestamp}.png"
                            save_table_as_image(vertical_bill_df, bill_img_path)
                            
                            # Download button for bill
                            st.download_button(
                                label=" Download Bill",
                                data=bill_df.to_csv(index=False),
                                file_name=bill_filename,
                                mime="text/csv"
                            )

                            # Display payment QR
                            payment_qr_path = "D:/projects @/smart patient medicine tracker/paymt.qr.jpg"
                            if os.path.exists(payment_qr_path):
                                st.write("### Scan QR Code to Pay")
                                st.image(payment_qr_path, width=300)
                else:
                    st.error("No QR code found in the image")
            except Exception as e:
                st.error(f"Error reading QR code: {str(e)}")

if __name__ == "__main__":
    main()
