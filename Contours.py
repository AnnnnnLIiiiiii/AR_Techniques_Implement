class Contours:
	def __init__(self, cnts, hierarchys):
		self.__epsilonRatio = 0.01
		self.__cnts = \
			[cv2.approxPolyDP(cnt,self.__epsilonRatio*cv2.arcLength(cnt,True),True) for cnt in cnts]
		self.__hierarchys = hierarchys
		self.amount = len(cnts)

	def getMember(self, index=None):
		if index != None:
			return self.__cnts[index]
		return self.__cnts

	def getHierarchy(self, index=None):
		if index != None:
			return self.__hierarchys[index]
		return self.__hierarchys

	def getCenter(self, index=None):
		cnt_moments = [cv2.moments(cnt) for cnt in self.__cnts]
		centers = [(int(M['m10']/M['m00']), int(M['m01']/M['m00'])) for M in cnt_moments]
		if index != None:
			return centers[index]
		return centers

def center(cnt):
	M = cv2.moments(cnt)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	return (cx, cy)

def targetContourExtract(contours, hierarchy):
	selected = []
	sel_hier = []
	for i in range(len(contours)):
		"selecte countour by the area and the hierarchy (no previous)"
		if (cv2.contourArea(contours[i]) >= 500) and ((hierarchy[0, i, 1] == -1)):
			selected.append(contours[i])
			sel_hier.append(hierarchy[0, i])
	selectCnts = Contours(selected, sel_hier)
	return selectCnts

def poseDetect(img):
	shirk_img = img[5:-5, 5:-5]
	v_half = int(shirk_img.shape[0]/2)
	h_half = int(shirk_img.shape[1]/2)
	x_0, y_0 = 0, 0
	quator_sum = np.array([[shirk_img[:v_half, :h_half].sum(), shirk_img[:v_half, h_half:].sum()],\
		[shirk_img[v_half:, :h_half].sum(), shirk_img[v_half:, h_half:].sum()]])

	"find white corner"
	corner_map = np.array([[0, 0], [img.shape[1]-1, 0], \
		[0, img.shape[0]-1], [img.shape[1]-1, img.shape[0]-1]])
	white_corner = corner_map[np.argmax(quator_sum)]

	"sort corners order in (x,y)"
	order = np.array([[0, 0], [img.shape[1]-1, 0], \
		[img.shape[1]-1, img.shape[0]-1], [0, img.shape[0]-1]])
	for i in range(len(order)):
		if sum(abs(order[i] - white_corner)) == 0:
			first = i
	corner_ordered = np.vstack((order[first:], order[:first]))
	corner_ordered = np.hstack((corner_ordered, np.ones((4, 1)))).astype(np.int)
	return corner_ordered

def transform(selectedCnts, template_shape, template_pts, img, lena=None):
	"define show window size"
	if selectedCnts.amount == 0:
		# print("cannot work with taht now")
		return []
	else:
		if selectedCnts.amount == 1:
			# if 4 <= (len(selectedCnts.getMember()[0])) < 8:
			# 	print("Outer Rectangle")
			# else:
			# 	print("Inner ar")
			return []
		else:
			if selectedCnts.amount == 2:
				cnt_pt_list = [len(cnt) for cnt in selectedCnts.getMember()]
				if sum(cnt_pt_list) <= 10:
					tagBlackPts = selectedCnts.getMember(-1)
				else:
					tagBlackPts = selectedCnts.getMember(np.argmin(cnt_pt_list))
			else:
				cnt_pt_list = [len(cnt) for cnt in selectedCnts.getMember()]
				for i in range(selectedCnts.amount):
					"no first child and no parent?"
					if (selectedCnts.getHierarchy(i)[2] != -1) and (selectedCnts.getHierarchy(i)[3] != -1):
						tagBlackPts = selectedCnts.getMember(i)
			x,y,w,h = cv2.boundingRect(tagBlackPts)
			crop_image = img[y:(y+h+1), x:(x+w+1)]

			"shrink tag coordinates to the size in corp image"
			tag_cord = tagBlackPts - np.array([[x, y]])
			crop_image = cv2.cvtColor(crop_image, cv2.COLOR_GRAY2BGR) #"sent binary image into cornerOrder function"
			# crop_image = cv2.circle(crop_image, tuple(tag_cord[0, 0].astype(np.int)), 3, (0,0,255), -1)
			M = cv2.getAffineTransform(tag_cord[:3].reshape(template_pts.shape).astype(np.float32), template_pts)
			dst = cv2.warpAffine(crop_image, M, template_shape)
			dst = cv2.GaussianBlur(dst,(5,5),0)

			"find correct order of corners"
			right_corners = poseDetect(dst[..., 0])
			dst = cv2.circle(dst, tuple(right_corners[0, :2]), 20, (0,0,255), -1)
			M_inv = cv2.invertAffineTransform(M)
			warped_corners = (right_corners @ M_inv.T).astype(np.int)
			cv2.imshow("Transform", dst)
			return warped_corners + np.array([[x, y]])

def replace(src, target, coords):
	"mask out template from src"
	mask = np.ones(src.shape).astype(np.uint8)
	mask = cv2.fillPoly(mask,[coords], (0,0,0))
	masked_src = np.array(mask * src)

	"get homography"
	rows, cols, _ = target.shape
	target_pts = np.float32([[0,0],[rows,0],[rows,cols]])
	M = cv2.getAffineTransform(target_pts, coords[:3].astype(np.float32))
	warped_target = cv2.warpAffine(target, M, (src.shape[1], src.shape[0]))
	dst = cv2.bitwise_or(masked_src, warped_target)

	return dst