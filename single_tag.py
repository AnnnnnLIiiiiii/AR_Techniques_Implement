import numpy as np
import cv2
from Contours import *

def main():
	"load templates"
	ar_template = cv2.imread('../data/Reference Images/ref_marker.png')
	rows, cols, _ = ar_template.shape
	template_shape = (cols, rows)
	template_pts = np.float32([[0,0],[rows,0],[rows,cols]])
	lena = cv2.imread('../data/Reference Images/Lena.png')

	"load video"
	cap = cv2.VideoCapture('../data/Input Sequences/Tag0.mp4')
	read_status, frame = cap.read()

	"define show window size"
	base_shape = frame.shape
	# cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
	# cv2.resizeWindow('Output', int(base_shape[1]*0.5), int(base_shape[0]*0.5))
	cv2.namedWindow('Output2', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Output2', int(base_shape[1]*0.5), int(base_shape[0]*0.5))

	while read_status == True:
		b_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		"appply Otsu's thresholding"
		blur = b_frame
		_,th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

		"contour finding"
		_, contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		selectCnts = targetContourExtract(contours, hierarchy)

		"re-finding contours by applying Gaussian filtering if no countour being selected"
		if selectCnts.amount == 0: 
			blur = cv2.GaussianBlur(b_frame,(5,5),0)
			_,th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			_, contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
			selectCnts = targetContourExtract(contours, hierarchy)

		th_rgb = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
		th_rgb = cv2.drawContours(th_rgb, selectCnts.getMember(), -1, (0,255,0), 2)

		tagPts = transform(selectCnts, template_shape, template_pts, th)
		if len(tagPts) != 0:
			warp_result = replace(frame, lena, tagPts)
		# cv2.imshow('Output', th_rgb)
		cv2.imshow('Output2', warp_result)
		cv2.waitKey(1)
		read_status, frame = cap.read()   

	cap.release()
	cv2.destroyAllWindows()

main()
