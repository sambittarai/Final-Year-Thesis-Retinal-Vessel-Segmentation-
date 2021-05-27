import cv2

def clahe_rgb(img_path, cliplimit=None, tilesize=8):
    
    """ 
    For RGB images, the image is first converted to the LAB format, 
    and then CLAHE is applied to the isolated L channel.
    Input:
      img_path: path to the image file
      cliplimit: the high contrast limit applied to CLAHE processing; this value is often 2, 3 or 4.
      tilesize: defines the local neighborhood for histogram equalization
    Returns:
      bgr: image after CLAHE processing
    """ 
    bgr = cv2.imread(img_path)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=cliplimit,tileGridSize=(tilesize, tilesize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr