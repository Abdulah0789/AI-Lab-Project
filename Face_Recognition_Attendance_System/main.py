import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from tkinter import *
from tkinter import filedialog, messagebox

#Images
students_dir = 'students_data'
os.makedirs(students_dir, exist_ok=True)

#CSV
attendance = 'attendance.csv'
frecognize = cv2.face.LBPHFaceRecognizer_create()
fcascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def is_registered(rollno):
    stud_file = os.listdir(students_dir)
    for file in stud_file:
        if file.split('_')[1].split('.')[0] == rollno.zfill(3):
            return True
    return False

def register_stud(name, rollno, image):
    if is_registered(rollno):
        messagebox.showerror("Error", f"Student with Roll No: {rollno} is already registered.")
        return False
    image = cv2.imread(image)
    if image is None:
        messagebox.showerror("Error", "Unable to read the image. Check the file path.")
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face = fcascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(face) == 0:
        messagebox.showerror("Error", "No face detected. Try again.")
        return False

# Check registered student
    if len(os.listdir(students_dir)) > 0:
        registered_faces = []
        labels = []
        for file in os.listdir(students_dir):
            stud_img = cv2.imread(os.path.join(students_dir, file), 0)
            if stud_img is not None:
                registered_faces.append(stud_img)
                labels.append(int(file.split('_')[1].split('.')[0]))

        frecognize.train(registered_faces, np.array(labels))

# Check if registered
        for (x, y, w, h) in face:
            face_region = gray[y:y + h, x:x + w]
            try:
                label, confidence = frecognize.predict(face_region)
                if confidence < 50:
                    messagebox.showerror("Error", f"This face is already registered with a different Roll No.")
                    return False
            except Exception as e:
                print(f"Error during face recognition: {e}")
                pass

# Save faces
    for (x, y, w, h) in face:
        face_region = gray[y:y + h, x:x + w]
        stud_file = os.path.join(students_dir, f"{name}_{rollno.zfill(3)}.jpg")
        cv2.imwrite(stud_file, face_region)
        messagebox.showinfo("Success", f"Student {name} (Roll No: {rollno}) registered successfully.")
        return True

    return False


def mark_attend(image):
    global attendance 
    # Time limit 9:00 AM
    now = datetime.now()
    curr_time = now.strftime("%H:%M")
    # if curr_time > "09:00":
    #     messagebox.showerror("Late", "You are late! Attendance cannot be marked after 9:00 AM.")
    #     return

# Create CSV
    if not os.path.exists(attendance):
        pd.DataFrame(columns=['Name', 'Roll No', 'Date', 'Time']).to_csv(attendance, index=False)

    image = cv2.imread(image)
    if image is None:
        messagebox.showerror("Error", "Unable to read the image. Check the file path.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = fcascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(face) == 0:
        messagebox.showerror("Error", "No face detected. Try again with a clearer image.")
        return

    registered_faces = []
    labels = []
    stud_file = []

    for file in os.listdir(students_dir):
        stud_img = cv2.imread(os.path.join(students_dir, file), 0)
        if stud_img is not None:
            registered_faces.append(stud_img)
            labels.append(int(file.split('_')[1].split('.')[0]))
            stud_file.append(file)

    if len(registered_faces) == 0:
        messagebox.showerror("Error", "No students registered yet.")
        return

# Training
    frecognize.train(registered_faces, np.array(labels))

# Check faces
    for (x, y, w, h) in face:
        face_region = gray[y:y + h, x:x + w]
        label, confidence = frecognize.predict(face_region)

        match_file = [file for file in stud_file if file.split('_')[1].split('.')[0] == str(label).zfill(3)]

        if match_file and confidence < 50:
            name = match_file[0].split('_')[0]
            rollno = match_file[0].split('_')[1].split('.')[0]
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

# Check marked
            attendance_df = pd.read_csv(attendance)
            if not ((attendance_df['Roll No'] == rollno) & (attendance_df['Date'] == date)).any():
                new_entry = pd.DataFrame({'Name': [name], 'Roll No': [rollno], 'Date': [date], 'Time': [time]})
                attendance_df = pd.concat([attendance_df, new_entry], ignore_index=True)
                attendance_df.to_csv(attendance, index=False)
                messagebox.showinfo("Success", f"Attendance marked for {name} (Roll No: {rollno})")
            else:
                messagebox.showinfo("Info", f"Attendance already marked for {name} today.")
            return

    messagebox.showerror("Error", "No matching student found.")


root = Tk()
root.title("Face Recognition Attendance System")
root.geometry("400x400")
Label(root, text="Student Name:").pack(pady=5)

name_entry = Entry(root, width=40)
name_entry.pack(pady=5)

Label(root, text="Roll Number:").pack(pady=5)
rollno_entry = Entry(root, width=40)
rollno_entry.pack(pady=5)

Button(root, text="Register Student", command=lambda: register_stud(name_entry.get(), rollno_entry.get(), filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]))).pack(pady=10)
Button(root, text="Mark Attendance", command=lambda: mark_attend(filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]))).pack(pady=10)
root.mainloop()




