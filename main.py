from keras_preprocessing import image

from model import *
from keras.models import load_model
from variable import *
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
from model_svm import *

my_w = tk.Tk()
my_w.geometry("550x400")  # Size of the window
my_w.title('Cat and Dogs')
my_font1=('times', 22, 'bold')
l1 = tk.Label(my_w,text='Add photo with cat or dog',width=30,font=my_font1)
l1.grid(row=1,column=1)
b1 = tk.Button(my_w, text='Upload File',
   width=20,command = lambda:upload_file())
label= tk.Label(my_w, image= '')
label.grid(row=3, column=1)
b1.grid(row=2,column=1)


def upload_file():
  global img
  f_types = [('Jpg Files', '*.jpg')]
  filename = filedialog.askopenfilename(filetypes=f_types)
  img = ImageTk.PhotoImage(file=filename)
 # b2 = tk.Button(my_w, image=img)  # using Button
 # b2.grid(row=3, column=1)
  label.configure(image=img)
  label.image = img
  predictImgCnn(filename)
  predictImgSvm(filename)

def predictImgCnn(filename):
  model.load_weights('model.h5')
  im = Image.open(filename)
  im = im.resize(Image_Size)
  im = np.expand_dims(im, axis=0)
  im = np.array(im)
  im = im / 255
  pred = model.predict([im])
  if (pred[0][1] >= 0.5):
    l3 = tk.Label(my_w, text='CNN: Dog '+ str(pred[0][1])+"/1", width=30, font=my_font1)
    l3.grid(row=4, column=1)
    print("Dog", pred[0][1])
  else:
    l3 = tk.Label(my_w, text='CNN: Cat '+ str(pred[0][0])+"/1", width=30, font=my_font1)
    l3.grid(row=4, column=1)
    print("Cat", pred[0][0])

def predictImgSvm(filename):
  test_image = image.load_img(filename, target_size=(64, 64))
  test_image = image.img_to_array(test_image)
  test_image = test_image / 255
  test_image = np.expand_dims(test_image, axis=0)
  svm.load_weights('model_svm.h5')
  result = svm.predict(test_image)
  if result[0] < 0:
    l4 = tk.Label(my_w, text='SVM: Cat '+ str(result[0][0]+3)+'/3', width=30, font=my_font1)
    l4.grid(row=5, column=1)
    print("SVM: The image classified is cat")
  else:
    l4 = tk.Label(my_w, text='SVM: Dog '+ str(result[0][0])+'/3', width = 30, font = my_font1)
    l4.grid(row=5, column=1)
    print("SVM: The image classified is dog")

my_w.mainloop()