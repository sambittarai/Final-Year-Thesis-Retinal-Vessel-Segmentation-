#=================================================================================================================================================
# This is an application which takes a single retinal image at a time and displays its preprocessed image and segmentation mask in the GUI viewer.
#=================================================================================================================================================

import torch
from models import UNetFamily
import torch.backends.cudnn as cudnn
import os
import argparse
from config import parse_args
from extract_patches import *
import numpy as np
from dataset import TestDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from pre_process_1 import my_PreProc
from pre_process_2 import clahe_rgb
from tkinter import *
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk, ImageOps
from fractal_dimension import fractal_dimension
import tkinter.font as font
import cv2
# progressbar - batch_idx

class Test():
	def __init__(self, args, test_img_path):
		self.args = args
		self.test_img_path = test_img_path
		assert (args.stride_height <= args.test_patch_height and args.stride_width <= args.test_patch_width)

		#Extract Patches
		self.patches_imgs_test, self.test_imgs, self.new_height, self.new_width = get_data_test_overlap(
			test_img_path=test_img_path,
			patch_height=args.test_patch_height,
			patch_width=args.test_patch_width,
			stride_height=args.stride_height,
			stride_width=args.stride_width
			)
		self.img_height =self.test_imgs.shape[2]
		self.img_width =self.test_imgs.shape[3]

		test_set = TestDataset(self.patches_imgs_test)
		self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

	def inference(self, net):
		net.eval()
		preds_outputs = []
		preds_prob_dist = []
		with torch.no_grad():
			for batch_idx, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
				#inputs = inputs.cuda() #if you are using GPU
				outputs = net(inputs)
				outputs_prob_dist = outputs[:,1].data.cpu().numpy() #probability distribution
				outputs_mask = outputs.argmax(dim = 1).data.numpy() #segmentation mask
				preds_prob_dist.append(outputs_prob_dist)
				preds_outputs.append(outputs_mask)
		predictions_mask = np.concatenate(preds_outputs, axis=0)
		predictions_prob_dist = np.concatenate(preds_prob_dist, axis=0)
		self.pred_patches_mask = np.expand_dims(predictions_mask,axis=1)
		self.pred_patches_prob = np.expand_dims(predictions_prob_dist,axis=1)

		return self.pred_patches_mask, self.pred_patches_prob

	def evaluate(self):
		self.pred_imgs_mask = recompone_overlap(
			self.pred_patches_mask, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)

		self.pred_imgs_mask = self.pred_imgs_mask[:, :, 0:self.img_height, 0:self.img_width]

		self.pred_imgs_prob_dist = recompone_overlap(
			self.pred_patches_prob, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)

		self.pred_imgs_prob_dist = self.pred_imgs_prob_dist[:, :, 0:self.img_height, 0:self.img_width]

		return self.pred_imgs_mask, self.pred_imgs_prob_dist


def run():
	torch.multiprocessing.freeze_support()

def predict(test_img_path, args):
	#device
	device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu") #device = cpu
	#Model Architecture
	net = UNetFamily.U_Net(args.in_channels, args.classes).to(device)
	cudnn.benchmark = True
	# Load checkpoint
	# print('==> Loading checkpoint...')
	checkpoint = torch.load(args.best_model_path, map_location=device)
	net.load_state_dict(checkpoint['net'])

	eval = Test(args, test_img_path)
	pred_patches_mask, pred_patches_prob_dist = eval.inference(net)
	pred_img_mask, pred_img_prob_dist = eval.evaluate()

	return pred_img_mask, pred_img_prob_dist

def show_image():
	global path
	path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image File", filetypes=(("TIF File", "*.tif"), ("JPG File", ".jpg"), ("PNG File", "*.png"), ("All Files", "*.*")))
	img = Image.open(path)
	img.thumbnail((300, 300))
	img = ImageTk.PhotoImage(img)
	label_1.configure(image=img)
	label_1.image = img

def show_preprocess_1():
	test_imgs = load_data(path)
	test_imgs = my_PreProc(test_imgs)
	test_imgs = test_imgs * 255.
	test_imgs = np.squeeze(test_imgs, 0)
	test_imgs = np.squeeze(test_imgs, 0)
	test_imgs = Image.fromarray(test_imgs).convert('RGB')
	test_imgs.thumbnail((300, 300))
	test_imgs = ImageTk.PhotoImage(test_imgs)
	label_2.configure(image=test_imgs)
	label_2.image = test_imgs

def show_preprocess_2():
	test_imgs = clahe_rgb(path, cliplimit=4, tilesize=16)
	#test_imgs = test_imgs * 255.
	test_imgs = Image.fromarray(test_imgs)
	test_imgs.thumbnail((300, 300))
	test_imgs = ImageTk.PhotoImage(test_imgs)
	label_6.configure(image=test_imgs)
	label_6.image = test_imgs

def show_combined():
	return

def calculate_f_d(pred_img_mask):
	pred_img_mask = pred_img_mask[0,0,:,:]
	# pred_img_mask = np.where(pred_img_mask > 0, 1, 0)
	f_d = fractal_dimension(pred_img_mask)
	return f_d

def show_f_d():
	label_5.configure(text="Fractal Dimension is: "+str(f_d), font=buttonFont)

def show_segmentation(args):
	global f_d

	pred_img_mask, pred_img_prob_dist = predict(path, args)
	pred_img_mask = np.where(pred_img_mask > 0, 1, 0)
	f_d = calculate_f_d(pred_img_mask)

	pred_img_mask = (255. * pred_img_mask[0,0,:,:]).astype(np.uint8)
	pred_img_prob_dist = (255. * pred_img_prob_dist[0,0,:,:]).astype(np.uint8)

	pred_img_mask = Image.fromarray(pred_img_mask).convert('RGB')
	pred_img_prob_dist = Image.fromarray(pred_img_prob_dist).convert('RGB')

	pred_img_mask.thumbnail((300, 300))
	pred_img_prob_dist.thumbnail((300, 300))

	pred_img_mask = ImageTk.PhotoImage(pred_img_mask)
	pred_img_prob_dist = ImageTk.PhotoImage(pred_img_prob_dist)

	label_3.configure(image=pred_img_mask)
	label_3.image = pred_img_mask

	label_4.configure(image=pred_img_prob_dist)
	label_4.image = pred_img_prob_dist


root = Tk()
root.title("Retinal Vessel Segmentation")
root.configure(bg='#d9fffb')
root.iconbitmap("G:/IIT_MADRAS_DD/Semesters/10th_sem/DDP_new_topic/My work/Code/Retinal_Vessel_Segmentation/GUI/icon.ico")
# root.geometry("850x500")
# root.resizable(False, False)
args = parse_args()

background_img = ImageTk.PhotoImage(
	Image.open("G:/IIT_MADRAS_DD/Semesters/10th_sem/DDP_new_topic/My work/Code/Retinal_Vessel_Segmentation/GUI/background_image.gif").resize((300, 300)))

label_1 = Label(root, image=background_img)
label_1.grid(row=0, column=0, padx=10)

buttonFont = font.Font(family='Helvetica', size=9, weight='bold')
button1 = Button(root, text="Browse Retinal Image", command=show_image, font=buttonFont)
button1.grid(row=1, column=0, pady=5)

label_2 = Label(root, image=background_img)
label_2.grid(row=0, column=1, padx=10)

button2 = Button(root, text="Preprocess 1", command=show_preprocess_1, font=buttonFont)
button2.grid(row=1, column=1, pady=5)

label_3 = Label(root, image=background_img)
label_3.grid(row=0, column=2, padx=10)

button3 = Button(root, text="Predict Segmentation Mask", command=lambda: show_segmentation(args), font=buttonFont)
button3.grid(row=1, column=2, pady=5)

label_4 = Label(root, image=background_img)
label_4.grid(row=0, column=3, padx=10)

label_5 = Label(root, text="")
label_5.grid(row=2, column=2, padx=10, pady=5)

button5 = Button(root, text="Fractal Dimension", command=show_f_d, font=buttonFont)
# button5 = Button(root, text="Fractal Dimension", font=buttonFont)
button5.grid(row=2, column=1, pady=5)

label_6 = Label(root, image=background_img)
label_6.grid(row=2, column=0, padx=10, pady=5)

button6 = Button(root, text="Preprocess 2", command=show_preprocess_2, font=buttonFont)
button6.grid(row=3, column=0, pady=5)

label_7 = Label(root, image=background_img)
label_7.grid(row=2, column=3, padx=10, pady=5)

button6 = Button(root, text="Display Overlap Image", command=show_combined, font=buttonFont)
button6.grid(row=3, column=3, pady=5)

root.mainloop()

'''
if __name__ == '__main__':
	# run()
	# test_img_path = "G:/IIT_MADRAS_DD/Semesters/10th_sem/DDP_new_topic/My work/Code/Retinal_Vessel_Segmentation/Datasets/CHASEDB/testing/images/Image_13L.jpg"
	test_img_path = "G:/IIT_MADRAS_DD/Semesters/10th_sem/DDP_new_topic/My work/Code/Retinal_Vessel_Segmentation/Datasets/DRIVE/test/images/18_test.tif"
	#test_mask_path = "G:/IIT_MADRAS_DD/Semesters/10th_sem/DDP_new_topic/My work/Code/Retinal_Vessel_Segmentation/Datasets/DRIVE/test/1st_manual/18_manual1.gif"

	args = parse_args()
	pred_img_mask, pred_img_prob_dist = predict(test_img_path, args)

	# im = (255. * pred_img_prob_dist[0,0,:,:]).astype(np.uint8)
	# im = np.where(im >= np.max(im)/4, 255, 0)
	# im = Image.fromarray(im).convert('RGB')
	# im.save("G:/IIT_MADRAS_DD/Semesters/10th_sem/DDP_new_topic/My work/Code/Retinal_Vessel_Segmentation/GUI/Image_Predictions/CHASEDB/13L/pred_prob_dist_13L.png")
	# dice = np.sum(2.0*pred2_bin*img) / (np.sum(pred2_bin) + np.sum(img))
'''

# m = (10*6 + 9*7 + 3*9 + 10*7 + 7*7 + 10*6) + (9*7 + 6*9 + 12*7 + 10*6 + 10*7 + 4*7) + (3*9 + 3*8) + 6*8 + 15*7 + 6*4 + 13*9 + 12*7 + 9*8 + 9*9 + 12*8 + 15*8 + 15*8 + 9*8 + 9*8 + 9*8 + 12*7 + 10*8 + 9*8 + 9*8 + 9*8 + 10*6 + 10*8 + 9*6 + 3*9 + 12*6 + 9*6 + 40*9 + 9*7
# n = 49 + 51 + 6 + 61 + 60 + 58 + 62 + 40 + 9
# m/n


# add a save button (next to browse button).