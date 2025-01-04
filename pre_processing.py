import cv2
import numpy as np
from scipy import ndimage

# make the image right oriented
def rotate_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # invert gray scale -> image white and the background black
    gray = cv2.bitwise_not(gray)

    # threshold the image, setting all foreground pixels to 255 and all background pixels to 0
    # to separate the text from the background.
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] # cv2.THRESH_OTSU -> automatic thresholding technique 


    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make it positive
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Apply rotation - INTER_CUBIC -> better quality
    rotated = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def remove_borders(cleaned_image,cleaned_image_inv):
    # edge detection - pertureSize = 3 -> size of sobel kernel
	edges = cv2.Canny(cleaned_image_inv,50,150,apertureSize = 3)
	minLineLength = 100
	maxLineGap = 10
    # Detect line in edge detection 
	lines = cv2.HoughLinesP(edges,1,np.pi/180,10,minLineLength,maxLineGap)
	if lines is None:
		return cleaned_image
	for line in lines:
        # first and end point
		x1,y1,x2,y2 =line[0]
        # daw black line - 2 is thickness 
		cv2.line(cleaned_image,(x1,y1),(x2,y2),(0,0,0),2)
        # draw white line - 2 is thickness 
		cv2.line(cleaned_image_inv,(x1,y1),(x2,y2),(255,255,255),2)

	return  cleaned_image

def remove_watermark(img):
	alpha = 2.0
	beta = -160
    # low brightness and hight contrast of the image
	new = alpha * img + beta
    #   Remove exceeding 
	new = np.clip(new, 0, 255).astype(np.uint8)
	return new


def pre_processing(image_path):
    image = cv2.imread(image_path)
    rotated = rotate_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # remove noise from gray scale - > h, hForColorComponents, templateWindowSize, and searchWindowSize
    image_without_noise=cv2.fastNlMeansDenoising(gray,6,6,7,21) 
    # 9-> block size , mean = 2
    img_clean = cv2.adaptiveThreshold(image_without_noise,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,9,2)
    cv2.imwrite(r"C:\Users\Lenovo\Downloads\poject\project\result_image\Result_image_clean.jpg",img_clean)
    return img_clean

