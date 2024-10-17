import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Button, Canvas
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

sys.stdout.reconfigure(encoding='utf-8')

model = ResNet50(weights='imagenet')

class ImageClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Classifier Using ResNet50")
        self.geometry("500x500")
        self.configure(bg='#F9F9F9')  

        self.heading = Label(self, text="Upload Image", font=("Arial", 18, "bold"), bg='#F9F9F9')
        self.heading.pack(pady=20)

        self.upload_frame = Canvas(self, width=200, height=200, bg='#EDEDED', bd=2, relief="groove")
        self.upload_frame.pack(pady=10)
        self.upload_frame.create_text(100, 100, text="Drag and Drop a file,\nor click to upload", font=("Arial", 12), justify="center")

        self.upload_frame.bind("<Button-1>", self.upload_image)

        self.upload_button = Button(self, text="Upload Image", command=self.upload_image, bg="#FF0000", fg="white", font=("Arial", 12, "bold"))
        self.upload_button.pack(pady=20)

        self.result_label = None

    def upload_image(self, event=None):
        """Handles image upload and performs classification"""
        
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        
        if file_path:
            
            img = Image.open(file_path)
            img = img.resize((200, 200), Image.Resampling.LANCZOS)  

            
            img_tk = ImageTk.PhotoImage(img)
            self.upload_frame.delete("all")  
            self.upload_frame.config(width=200, height=200)  
            self.upload_frame.create_image(100, 100, image=img_tk)  
            self.upload_frame.image = img_tk  

            
            prediction = self.classify_image(file_path)

            
            self.display_result(prediction)
        else:
            messagebox.showwarning("No File Selected", "You did not select any file.")

    def classify_image(self, image_path):
        """Classifies the given image using ResNet50 model"""
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)  
        img_array = np.expand_dims(img_array, axis=0)  
        img_array = preprocess_input(img_array)  

        predictions = model.predict(img_array)

        decoded_predictions = decode_predictions(predictions, top=1)[0]
        return f"{decoded_predictions[0][1]} ({decoded_predictions[0][2]*100:.2f}%)"  

    def display_result(self, result):
        """Displays the classification result"""
        if self.result_label is None:
            self.result_label = Label(self, text=result, font=("Arial", 12), bg='#F9F9F9')
            self.result_label.pack(pady=20)
        else:
            self.result_label.configure(text=result)

if __name__ == "__main__":
    app = ImageClassifierApp()
    app.mainloop()
