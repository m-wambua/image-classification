from tkinter import *
import tkinter as tk

# loading Python Imaging Library
from PIL import ImageTk, Image

# To get the dialog box to open when required
from tkinter import filedialog

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


def classify(img_path):
    
    
    img=image.load_img(img_path,target_size=(224,224))
    model=tf.keras.applications.resnet50.ResNet50()
    img_array=image.img_to_array(img)
    img_batch=np.expand_dims(img_array,axis=0)
    img_preprocessed=preprocess_input(img_batch)
    prediction=model.predict(img_preprocessed)
    
    results= decode_predictions(prediction,top=3)[0]
    return results
def select_image():
    root.filename = filedialog.askopenfilename(initialdir = "/", title = "Select file", filetypes = (("jpeg files", "*.jpg"), ("all files", "*.*")))
    print(root.filename)
    # classify the selected image
    results = classify(root.filename)

    # Clear previous labels
    for widget in frame.winfo_children():
        widget.destroy()

    # Open image using PIL
    img = Image.open(root.filename)
    img = img.resize((250,250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    image_label = tk.Label(frame, image=img)
    image_label.image = img
    image_label.pack()

    # Display the results on the GUI
    for result in results:
        label = tk.Label(frame, text=result[1])
        label.pack()

root = tk.Tk()
frame = tk.Frame(root)
frame.pack()

select_image_button = tk.Button(root, text="Select Image", command=select_image)
select_image_button.pack()

root.mainloop()


