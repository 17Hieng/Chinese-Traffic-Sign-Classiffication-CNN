# Import necessary library
import tkinter as tk
import PIL
import cv2
import csv
import numpy as np
from keras.models import load_model
from PIL import ImageTk
from tkinter import *
from tkinter import filedialog
from tqdm import tqdm
from skimage import exposure

# Load Trained Model
model = load_model('Model.h5')

# Map Sign Name to Classes
labels_dict = None

with open('mapSignnamesToClass.csv', mode='r') as infile:
    reader = csv.reader(infile)
    next(reader, None)
    labels_dict = {int(rows[0]): rows[1] for rows in reader}


# Normalization function
def normalize(image_data):
    '''Contrast Limited Adaptive Histogram Equalization (CLAHE). In addition to regular normalization,
    this function provides local contrast enhancement -- i.e., details of the image can be
    enhanced even in regions that are darker or lighter than most of the image.
    http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist
    '''

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
    image = PIL.Image.fromarray(image, 'RGB')
    image = image.resize((30, 30))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = image / 255
    image = normalize(image)
    pred = model.predict(image)
    result = pred.argmax()
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
        uploaded = PIL.Image.open(file_path)
        uploaded = uploaded.resize((400, 300))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload_btn = Button(btn_frame, text="Insert Image",
                    command=upload_image)
upload_btn.configure(background=BACKGROUND_COLOR,
                     foreground=FOREGROUND_COLOR, font=('Courier New', 12, 'bold'))
upload_btn.pack(side=BOTTOM)

classify_btn = Button(classify_btn_frame,
                      text="Classify Sign")
classify_btn.configure(background=BACKGROUND_COLOR,
                       foreground=FOREGROUND_COLOR, font=('Courier New', 12, 'bold'))

top.mainloop()