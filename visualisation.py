import argparse
import tkinter as tk
from options.train_options import TrainOptions
from models import create_model
from tkinter import *
from util.visualizer import save_images
from torchvision.transforms import ToTensor
from util import util
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from tkinter import colorchooser


from PIL import Image, ImageTk

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        # self.pack(fill=BOTH, expand=1)


def select_photos():
    to_visualize = ['gray', 'real', 'fake_reg']
    opt = TrainOptions().parse()
    opt.load_model = True
    opt.num_threads = 1
    opt.batch_size = 1
    opt.display_id = -1
    opt.phase = 'val'

    opt.serial_batches = True
    opt.aspect_ratio = 1.

    model = create_model(opt)
    model.setup(opt)

    # tensor_image = tensor_image.cuda()
    sample_ps = 0.125
    data_raw = [None] * 4

    # loader = transforms.Compose(transforms.ToTensor())
    # raw_image = Image.open(args.input)
    # # tensor_image = TF.to_tensor(raw_image)
    # tensor_image = loader(raw_image.float())
    # tensor_image = raw_image.unsqueeze(0)
    # data_raw[0] = tensor_image.cuda()

    # print(opt.input)
    tensor_image = Image.open(opt.input).convert('RGB')
    tensor_image = tensor_image.resize((opt.loadSize,opt.loadSize))
    tensor_image = ToTensor()(tensor_image).unsqueeze(0)
    data_raw[0] = tensor_image.cuda()
    # data_raw[0] = util.crop_mult(data_raw[0], mult=8)

    print(tensor_image.shape)
    # this also might be the error
    data = util.get_colorization_data(data_raw, opt, ab_thresh=0., p=sample_ps)
    print(data['A'].shape)
    print(data['B'].shape)
    print(data['hint_B'].shape)
    print(data['mask_B'].shape)

    # this is potentially where the error is, should set all the data
    model.set_input(data)

    # model.eval()
    model.test(True)

    # gets the visuals from the model
    visuals = util.get_subset_dict(model.get_current_visuals(), to_visualize)

    # output images
    raw_image = Image.fromarray(util.tensor2im(visuals['real']))
    image = raw_image.resize((450, 450), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(image)
    label = tk.Label(text= "Original Image",compound='top',image=image)
    label.photo = image  # assign to class variable to resolve problem with bug in `PhotoImage`
    label.grid(row=1, column=1)
    all_labels.append(label)

    raw_image = util.tensor2im(visuals['fake_reg']) / 255
    lab_raw_image = util.rgb2lab(raw_image, opt)

    image = raw_image.resize((450, 450), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(image)
    label = tk.Label(image=image,text= "Colourised Image",compound='top')
    label.bind("<Button-1>",lambda e:choose_colour())
    label.photo = image
    label.grid(row=1, column=3)

    print(util.tensor2im(visuals['fake_reg']))

    # label = tk.Label(text="PSNR Torch: "  & util.calculate_psnr_torch(original,util.tensor2im(visuals['fake_reg'])))
    # label.grid(row=4, column=2)
    original = Image.open(opt.input)
    original = original.resize((opt.loadSize,opt.loadSize))
    original = np.asarray(original)
    label = tk.Label(text="PSNR Numpy: "  + str(util.calculate_psnr_np(original,util.tensor2im(visuals['fake_reg']))))
    label.grid(row=4, column=1)

    label = tk.Label(text="MSE : " + str(util.calculate_mse(original, util.tensor2im(visuals['fake_reg']))))
    label.grid(row=4, column=3)

    label = tk.Label(text="MSE : " + str(util.calculate_mae(original, util.tensor2im(visuals['fake_reg']))))
    label.grid(row=4, column=2)

    all_labels.append(label)

def choose_colour () :
    color_code = colorchooser.askcolor(title ="Choose color")

if __name__ == '__main__':
    all_labels = []
    root = Tk()
    app = Window(root)
    label = tk.Label(text= "                   ")
    label.grid(row=0,column=0)
    label = tk.Label(text= "                   ")
    label.grid(row=0,column=0)
    label = tk.Label(text= "Press on the Original Image to add a colour 'HINT'")
    label.grid(row=1,column=2)
    button2 = tk.Button(root, text="Load Image and Colourise", command=select_photos)
    button2.grid(row=3, column=2)
    colourButton = tk.Button(root, text="Choose colour 'HINT'", )

    # root.wm_title("Tkinter window")
    root.geometry("1000x500")
    root.mainloop()

