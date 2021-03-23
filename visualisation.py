import argparse
import tkinter as tk
from options.train_options import TrainOptions
from models import create_model
from tkinter import *
from util.visualizer import save_images
from torchvision.transforms import ToTensor
from util import util
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms

# pip install pillow
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
    # opt.dataroot = '/data/cifar10png/test'
    opt.serial_batches = True
    opt.aspect_ratio = 1.

    model = create_model(opt)
    model.setup(opt)

    # tensor_image = tensor_image.cuda()
    sample_ps = 0.031253
    data_raw = [None] * 10

    # loader = transforms.Compose(transforms.ToTensor())
    # raw_image = Image.open(args.input)
    # # tensor_image = TF.to_tensor(raw_image)
    # tensor_image = loader(raw_image.float())
    # tensor_image = raw_image.unsqueeze(0)
    # data_raw[0] = tensor_image.cuda()

    print(args.input)
    tensor_image = Image.open(args.input).convert('RGB')
    tensor_image = ToTensor()(tensor_image).unsqueeze(0)
    data_raw[0] = tensor_image.cuda()

    # this also might be the error
    data = util.get_colorization_data(data_raw, opt, ab_thresh=0., p=sample_ps)

    # this is potentially where the error is, should set all the data
    model.set_input(data)

    # model.eval()
    model.optimize_parameters()

    # gets the visuals from the model
    visuals = util.get_subset_dict(model.get_current_visuals(), to_visualize)

    # output images
    raw_image = Image.open(Image.fromarray(util.tensor2im(visuals['real'])))
    image = raw_image.resize((450, 450), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(image)
    label = tk.Label(image=image)
    label.photo = image  # assign to class variable to resolve problem with bug in `PhotoImage`
    label.grid(row=1, column=1)
    all_labels.append(label)

    raw_image = Image.open(Image.fromarray(util.tensor2im(visuals['fake_reg'])))
    image = raw_image.resize((450, 450), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(image)
    label = tk.Label(image=image)
    label.photo = image
    label.grid(row=1, column=2)
    all_labels.append(label)


if __name__ == '__main__':
    args = TrainOptions().parse()
    all_labels = []
    root = Tk()
    app = Window(root)

    button2 = tk.Button(root, text="Load Image and Colourise", command=select_photos)
    button2.grid(row=0, column=0)

    # root.wm_title("Tkinter window")
    root.geometry("1000x500")
    root.mainloop()
