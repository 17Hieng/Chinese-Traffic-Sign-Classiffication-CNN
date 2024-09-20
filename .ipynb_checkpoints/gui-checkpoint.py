# Import necessary libraries
import tkinter as tk
import torch
import cv2
import csv
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from tqdm import tqdm
from skimage import exposure

# Load PyTorch Model
model = torch.load('model_full.pth')
model.eval()

# Map Sign Name to Classes
labels_dict = None

with open('mapSignnamesToClass.csv', mode='r') as infile:
    reader = csv.reader(infile)
    next(reader, None)
    labels_dict = {int(rows[0]): rows[1] for rows in reader}


# Normalization function
def normalize(image_data):
    '''Contrast Limited Adaptive Histogram Equalization (CLAHE).
    This function provides local contrast enhancement.'''
    norm = np.array([exposure.equalize_adapthist(image, clip_limit=0.1)
                     for image in tqdm(image_data)])
    return norm


# User Interface
BACKGROUND_COLOR = '#FFF'
FOREGROUND_COLOR = '#000'

# Interface
top = tk.Tk()
top.geometry('800x600')
top.title('Traffic Sign Recognition System by Group 1')
top.configure(background=BACKGROUND_COLOR)

# Heading
heading = Label(top, text="TRAFFIC SIGN CLASSIFICATION",
                pady=20, font=('MS Sans Serif', 22, 'bold'))
heading.configure(background=BACKGROUND_COLOR, foreground=FOREGROUND_COLOR)
heading.pack()

# Frame for button section
btn_frame = Frame(top, background=BACKGROUND_COLOR)
btn_frame.pack()
btn_frame.place(anchor='e', relx=0.95, rely=0.5)

classify_btn_frame = Frame(btn_frame, background=BACKGROUND_COLOR)
classify_btn_frame.pack(side=TOP)

# Frame for Image section
image_frame = Frame(top, width=400, height=400, background=BACKGROUND_COLOR)
image_frame.pack()
image_frame.place(anchor='w', relx=0.15, rely=0.5)

sign_image = Label(image_frame, background=BACKGROUND_COLOR)
sign_image.pack(side=TOP, pady=10)

label = Label(image_frame, background=BACKGROUND_COLOR,
              foreground=FOREGROUND_COLOR, font=('MS Sans Serif', 17))
label.pack(side=BOTTOM)


# Predict the class
def classify(file_path):
    global label_packed
    image = cv2.imread(file_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((30, 30))  # Resize to the input size of the model
    image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = normalize([image])[0]
    image = np.transpose(image, (2, 0, 1))  # Change to channel-first format for PyTorch
    image = torch.tensor(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1)
        result = pred.item()
        sign = labels_dict[result]
        print(sign)
        label.configure(text=sign)


# Show classify button when image is uploaded
def show_classify_button(file_path):
    classify_btn.configure(command=lambda: classify(file_path))
    classify_btn.pack(side=TOP)


# Upload Image Function
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded = uploaded.resize((400, 300))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        pass


upload_btn = Button(btn_frame, text="Insert Image", command=upload_image)
upload_btn.configure(background=BACKGROUND_COLOR,
                     foreground=FOREGROUND_COLOR, font=('Courier New', 12, 'bold'))
upload_btn.pack(side=BOTTOM)

classify_btn = Button(classify_btn_frame, text="Classify Sign")
classify_btn.configure(background=BACKGROUND_COLOR,
                       foreground=FOREGROUND_COLOR, font=('Courier New', 12, 'bold'))

top.mainloop()