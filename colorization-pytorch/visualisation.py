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
from PIL import Image, ImageCms
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
    to_visualize = ['gray', 'real', 'fake_reg', 'mask', 'hint']
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
    sample_ps = 0.03125
    data_raw = [None] * 4

    # loader = transforms.Compose(transforms.ToTensor())
    # raw_image = Image.open(args.input)
    # # tensor_image = TF.to_tensor(raw_image)
    # tensor_image = loader(raw_image.float())
    # tensor_image = raw_image.unsqueeze(0)
    # data_raw[0] = tensor_image.cuda()

    # print(opt.input)
    tensor_image = Image.open(opt.input).convert('RGB')
    tensor_image = tensor_image.resize((opt.loadSize, opt.loadSize))
    tensor_image = ToTensor()(tensor_image).unsqueeze(0)
    data_raw[0] = tensor_image.cuda()
    # data_raw[0] = util.crop_mult(data_raw[0], mult=8)

    data = util.get_colorization_data(data_raw, opt, ab_thresh=0., p=sample_ps)

    model.set_input(data)

    # model.eval()
    model.test(True)

    # gets the visuals from the model
    global visuals
    visuals = util.get_subset_dict(model.get_current_visuals(), to_visualize)

    # output images
    raw_image = Image.fromarray(util.tensor2im(visuals['real']))
    image = raw_image.resize((512, 512), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(image)
    label = tk.Label(text="Original Image", compound='top', image=image)
    label.photo = image  # assign to class variable to resolve problem with bug in `PhotoImage`
    label.grid(row=1, column=1)
    all_labels.append(label)

    raw_image = Image.fromarray(util.tensor2im(visuals['hint']))
    # lab_raw_image = util.rgb2lab(raw_image, opt)

    image = raw_image.resize((512, 512), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(image)
    label = tk.Label(image=image, text="Hint Image", compound='top')
    label.bind("<Button-1>", lambda e: choose_colour())
    label.photo = image
    label.grid(row=1, column=3)
    all_labels.append(label)

    raw_image = Image.fromarray(util.tensor2im(visuals['fake_reg']))
    image = raw_image.resize((512, 512), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(image)
    label = tk.Label(image=image, text="Colourised Image", compound='top')
    label.bind("<Button-1>", lambda e: choose_colour())
    label.photo = image
    label.grid(row=1, column=5)
    all_labels.append(label)

    original = Image.open(opt.input)
    original = original.resize((opt.loadSize, opt.loadSize))
    original = np.asarray(original)
    label = tk.Label(text="PSNR Numpy: " + str(util.calculate_psnr_np(original, util.tensor2im(visuals['fake_reg']))))
    label.grid(row=4, column=1)
    all_labels.append(label)

    label = tk.Label(text="MSE : " + str(util.calculate_mse(original, util.tensor2im(visuals['fake_reg']))))
    label.grid(row=4, column=3)
    all_labels.append(label)

    label = tk.Label(text="MSE : " + str(util.calculate_mae(original, util.tensor2im(visuals['fake_reg']))))
    label.grid(row=4, column=5)

    all_labels.append(label)


def choose_colour():
    global x_public, y_public
    color_code = colorchooser.askcolor(title="Choose color")
    # print("colour code")
    # print(color_code[0])

    num = 0
    RGB = [0, 0, 0]

    # rgb to xyz to lab conversion
    for value in color_code[0]:
        value = float(value) / 255
        if value > 0.04045:
            value = ((value + 0.055) / 1.055) ** 2.4
        else:
            value = value / 12.92
        RGB[num] = value * 100
        num = num + 1
    XYZ = [0, 0, 0, ]
    X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
    Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
    Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
    XYZ[0] = round(X, 4)
    XYZ[1] = round(Y, 4)
    XYZ[2] = round(Z, 4)
    XYZ[0] = float(XYZ[0]) / 95.047  # ref_X =  95.047   Observer= 2°, Illuminant= D65
    XYZ[1] = float(XYZ[1]) / 100.0  # ref_Y = 100.000
    XYZ[2] = float(XYZ[2]) / 108.883  # ref_Z = 108.883
    num = 0
    for value in XYZ:
        if value > 0.008856:
            value = value ** (0.3333333333333333)
        else:
            value = (7.787 * value) + (16 / 116)
        XYZ[num] = value
        num = num + 1

    Lab = [0, 0, 0]
    L = (116 * XYZ[1]) - 16
    a = 500 * (XYZ[0] - XYZ[1])
    b = 200 * (XYZ[1] - XYZ[2])
    Lab[0] = round(L, 4)
    Lab[1] = round(a, 4)
    Lab[2] = round(b, 4)



    mask = visuals['mask'].clone()
    fake_reg = visuals['fake_reg'].clone()

    print(fake_reg.shape)
    x_public = int(x_public / 2)
    y_public = int(y_public / 2)
    print(x_public, y_public)

    # fake_reg, mask = util.add_color_patch(data_raw, mask, opt, )
    fake_reg, mask = util.add_color_patch(visuals['fake_reg'], mask, opt, P=10, hw=[y_public, x_public], ab=[Lab[1], Lab[2]])

    raw_image = Image.fromarray(util.tensor2im(fake_reg))
    mask = Image.fromarray(util.tensor2im(visuals['hint']))
    update_images(raw_image,mask)


def update_images(img,mask):
    image = img.resize((512, 512), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(image)
    colourised_label = tk.Label(text="Original Image", compound='top', image=image)
    colourised_label.photo = image  # assign to class variable to resolve problem with bug in `PhotoImage`
    colourised_label.grid(row=1, column=5)
    all_labels.append(colourised_label)

    mask = mask.resize((512, 512), Image.ANTIALIAS)
    mask = ImageTk.PhotoImage(mask)
    colourised_label = tk.Label(text="Hint Image", compound='top', image=mask)
    colourised_label.photo = mask  # assign to class variable to resolve problem with bug in `PhotoImage`
    colourised_label.bind("<Button-1>", lambda e: choose_colour())
    colourised_label.grid(row=1, column=3)
    all_labels.append(colourised_label)
    root.update_idletasks()


def motion(event):
    global x_public, y_public
    x_public, y_public = event.x, event.y
    # print('{}, {}'.format(x_public, y_public))


if __name__ == '__main__':
    all_labels = []
    opt = TrainOptions().parse()
    root = Tk()
    app = Window(root)
    label = tk.Label(text="                   ")
    label.grid(row=0, column=0)
    label = tk.Label(text="                   ")
    label.grid(row=0, column=0)
    label = tk.Label(text="Press on the Original Image to add a colour 'HINT'")
    label.grid(row=1, column=2)
    label = tk.Label(text="Press on the Original Image to add a colour 'HINT'")
    label.grid(row=1, column=4)
    button2 = tk.Button(root, text="Load Image and Colourise", command=select_photos)
    button2.grid(row=3, column=3)

    root.wm_title("Colourisation PyTorch GUI")
    root.geometry("2300x610")
    root.bind('<Motion>', motion)
    root.mainloop()
