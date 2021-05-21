"""
This part mainly contains functions related to extracting image patches.
The image patches are randomly extracted in the fov(optional) during the training phase, 
and the test phase needs to be spliced after splitting
"""
import numpy as np
import random
from pre_process import my_PreProc
from PIL import Image


def load_data(test_img_path):
    '''
    Load the original image, grroundtruth and FOV of the data set in order, and check the dimensions
    # This function is up to date and working fine for the GUI 
    '''
    img = np.asarray(Image.open(test_img_path))
    img = np.expand_dims(img,0)
    img = np.transpose(img,(0,3,1,2))
    # img is of size (1, 3, H, W), (1, 1, H, W) with np.min = 0, np.max = 255
    return img


# =============================Load test data==========================================
def get_data_test_overlap(test_img_path, patch_height, patch_width, stride_height, stride_width):
    '''
    test_img_path, test_mask_path, test_FOV_path, patch_height, patch_width, stride_height, stride_width
    Load the original data and return the extracted patches for testing, return the ground truth in its original shape
    # This function is up to date and working fine for the GUI 
    '''
    # test_imgs_original - (1, 1, H, W), np.min = 0, np.max = 1 (for image, mask, FOV)
    test_imgs_original = load_data(test_img_path)
    test_imgs = my_PreProc(test_imgs_original)
    #extend the image so that it can be divided exactly by the patches dimensions
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)
    #extract the test patches from the all test images
    patches_imgs_test = extract_ordered_overlap(test_imgs,patch_height,patch_width,stride_height,stride_width)
    return patches_imgs_test, test_imgs_original, test_imgs.shape[2], test_imgs.shape[3]

def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    '''
    extend both images and masks so they can be divided exactly by the patches dimensions
    # This function is up to date and working fine for the GUI
    '''
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  # check the channel is 1 or 3
    img_h = full_imgs.shape[2]  #height of the image
    img_w = full_imgs.shape[3] #width of the image
    leftover_h = (img_h-patch_h)%stride_h  #leftover on the h dim
    leftover_w = (img_w-patch_w)%stride_w  #leftover on the w dim
    if (leftover_h != 0):  #change dimension of img_h
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],img_h+(stride_h-leftover_h),img_w))
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:img_h,0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    if (leftover_w != 0):   #change dimension of img_w
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],full_imgs.shape[2],img_w+(stride_w - leftover_w)))
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:full_imgs.shape[2],0:img_w] = full_imgs
        full_imgs = tmp_full_imgs

    #print("new padded images shape: " +str(full_imgs.shape))
    return full_imgs

def extract_ordered_overlap(full_imgs, patch_h, patch_w,stride_h,stride_w):
    '''
    Extract test image patches in order and overlap
    # This function is up to date and working fine for the GUI 
    '''
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    assert ((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  #// --> division between integers
    N_patches_tot = N_patches_img*full_imgs.shape[0]

    patches = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                patch = full_imgs[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
                patches[iter_tot]=patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    return patches  #array with all the full_imgs divided in patches

def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
    '''
    recompone the prediction result patches to images
    '''
    assert (len(preds.shape)==4)  #4D arrays
    assert (preds.shape[1]==1 or preds.shape[1]==3)  #check the channel is 1 or 3
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_img = N_patches_h * N_patches_w
    assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
    full_prob = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))
    full_sum = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))

    k = 0 #iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                full_prob[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=preds[k] # Accumulate predicted values
                full_sum[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=1  # Accumulate the number of predictions
                k+=1
    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0) 
    final_avg = full_prob/full_sum # Take the average
    assert(np.max(final_avg)<=1.0) # max value for a pixel is 1.0
    assert(np.min(final_avg)>=0.0) # min value for a pixel is 0.0
    return final_avg

# def pred_only_in_FOV(data_imgs,data_masks,FOVs):
#     '''
#     return only the predicted pixels contained in the FOV, for both images and masks
#     '''
#     assert (len(data_imgs.shape)==4 and len(data_masks.shape)==4)  #4D arrays
#     height = data_imgs.shape[2]
#     width = data_imgs.shape[3]
#     new_pred_imgs = []
#     new_pred_masks = []
#     for i in range(data_imgs.shape[0]):  #loop over the all test images
#         for x in range(width):
#             for y in range(height):
#                 if pixel_inside_FOV(i,x,y,FOVs):
#                     new_pred_imgs.append(data_imgs[i,:,y,x])
#                     new_pred_masks.append(data_masks[i,:,y,x])
#     new_pred_imgs = np.asarray(new_pred_imgs)
#     new_pred_masks = np.asarray(new_pred_masks)
#     return new_pred_imgs, new_pred_masks

# def kill_border(data, FOVs):
#     '''
#     Set the pixel value outside FOV to 0, only for visualization
#     '''
#     assert (len(data.shape)==4)  #4D arrays
#     assert (data.shape[1]==1 or data.shape[1]==3)  #check the channel is 1 or 3
#     height = data.shape[2]
#     width = data.shape[3]
#     for i in range(data.shape[0]):  #loop over the full images
#         for x in range(width):
#             for y in range(height):
#                 if not pixel_inside_FOV(i,x,y,FOVs):
#                     data[i,:,y,x]=0.0

# def pixel_inside_FOV(i, x, y, FOVs):
#     '''
#     function to judge pixel(x,y) in FOV or not
#     '''
#     assert (len(FOVs.shape)==4)  #4D arrays
#     assert (FOVs.shape[1]==1)
#     if (x >= FOVs.shape[3] or y >= FOVs.shape[2]): # Pixel position is out of range
#         return False
#     return FOVs[i,0,y,x]>0 #0==black pixels