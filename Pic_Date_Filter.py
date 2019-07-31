import os, sys
from PIL import Image
from matplotlib import pyplot as PLT
import pytesseract
import argparse
import numpy as np
import pytesseract
import numpy as np
import cv2
import imutils
from skimage import exposure
from pytesseract import image_to_string
import Crop

def extractor(imgPath)
	Im = Image.open(imgPath)  # your image
	width = Im.size[0]  # define W and H
	height = Im.size[1]
	Im = Im.crop((width * 0.78, height * 0.83, width * 0.92, height * 0.94))  # By Data Stamp Location in %
	NewIm = Im
	photo = Im.load()
	NewPhoto = NewIm.load()
	# photo = photo.convert('RGB')

	width = Im.size[0]  # define W and H
	height = Im.size[1]

	for y in range(0, height):  # each pixel has coordinates
		row = ""
		for x in range(0, width):

			RGB = photo[x, y]
			R, G, B = RGB  # now you can use the RGB value
			# print(R,G,B)
			if (R > 80 and G < 50 and B < 50) or (
					(R > 140) and ((R - G > 20) or G - R > 20) and (R - B > 20 or B - R > 20)):
				NewPhoto[x, y] = (0, 0, 0)
				# Bolding the Img
				try:
					for X in range(0, 3):
						for Y in range(0, 3):
							NewPhoto[x - 3 + X, y - 3 + Y] = (0, 0, 0)
				except:
					NewPhoto[x, y] = (0, 0, 0)

			else:
				NewPhoto[x, y] = (255, 255, 255)

	###################Crop###################

	photo = NewIm.load()
	width = NewIm.size[0]  # define W and H
	height = NewIm.size[1]
	cropSizeUp = 0
	cropSizeDown = 0
	flag = True
	for y in range(0, height):  # each pixel has coordinates
		for x in range(0, width):
			if (photo[x, y] == (0, 0, 0)):
				flag = False
				break
		if (flag):
			cropSizeUp = cropSizeUp + 1
		else:
			break
	flag = True
	for y in range(0, height):  # each pixel has coordinates
		for x in range(0, width):
			if (photo[x, height - y - 1] == (0, 0, 0)):
				flag = False
				break
		if (flag):
			cropSizeDown = cropSizeDown + 1
		else:
			break

	NewIm = NewIm.crop((0, cropSizeUp - 12, width, height - cropSizeDown + 12))  # By Data Stamp Location in %
	NewIm.save('test.jpg')

	#####################CONVERT 2 BMP################################
	ary = np.array(NewIm)

	# Split the three channels
	r, g, b = np.split(ary, 3, axis=2)
	r = r.reshape(-1)
	g = r.reshape(-1)
	b = r.reshape(-1)
	# Standard RGB to grayscale
	bitmap = list(map(lambda x: 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2], zip(r, g, b)))
	bitmap = np.array(bitmap).reshape([ary.shape[0], ary.shape[1]])
	bitmap = np.dot((bitmap > 128).astype(float), 255)
	NewIm = Image.fromarray(bitmap.astype(np.uint8))
	NewIm.save('Out.bmp')

	img = cv2.resize(cv2.imread('Out.bmp'), (660, 300))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return remove_noise(gray, 1)

def remove_noise(gray, num):
	Y, X = gray.shape
	nearest_neigbours = [[
		np.argmax(
			np.bincount(
				gray[max(i - num, 0):min(i + num, Y), max(j - num, 0):min(j + num, X)].ravel()))
		for j in range(X)] for i in range(Y)]
	result = np.array(nearest_neigbours, dtype=np.uint8)
	cv2.imwrite('Out1.bmp', result)
	return result


DIGITS_LOOKUP = {
	(1, 1, 1, 1, 1, 1, 0): 0,
	(1, 1, 0, 0, 0, 0, 0): 1,
	(1, 0, 1, 1, 0, 1, 1): 2,
	(1, 1, 1, 0, 0, 1, 1): 3,
	(1, 1, 0, 0, 1, 0, 1): 4,
	(0, 1, 1, 0, 1, 1, 1): 5,
	(0, 1, 1, 1, 1, 1, 1): 6,
	(1, 1, 0, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 0, 1, 1, 1): 9,
	(0, 0, 0, 0, 0, 1, 1): '-'
}
H_W_Ratio = 3.87
THRESHOLD = 10
arc_tan_theta = 6.0  # 数码管倾斜角度


def load_image(path, show=False):
	# todo: crop image and clear dc and ac signal
	gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	h, w = gray_img.shape
	# crop_y0 = 0 if h <= crop_y0_init else crop_y0_init
	# crop_y1 = h if h <= crop_y1_init else crop_y1_init
	# crop_x0 = 0 if w <= crop_x0_init else crop_x0_init
	# crop_x1 = w if w <= crop_x1_init else crop_x1_init
	# gray_img = gray_img[crop_y0:crop_y1, crop_x0:crop_x1]
	blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
	if show:
		cv2.imshow('gray_img', gray_img)
		cv2.imshow('blurred_img', blurred)
	return blurred, gray_img


def preprocess(img, threshold, show=False, kernel_size=(5, 5)):
	# 直方图局部均衡化
	clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
	img = clahe.apply(img)
	# 自适应阈值二值化
	dst = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 127, threshold)
	# 闭运算开运算
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
	dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
	dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)

	if show:
		cv2.imshow('equlizeHist', img)
		cv2.imshow('threshold', dst)
	return dst


def helper_extract(one_d_array, threshold=20):
	res = []
	flag = 0
	temp = 0
	for i in range(len(one_d_array)):
		if one_d_array[i] < 12 * 255:
			if flag > threshold:
				start = i - flag
				end = i
				temp = end
				if end - start > 20:
					res.append((start, end))
			flag = 0
		else:
			flag += 1

	else:
		if flag > threshold:
			start = temp
			end = len(one_d_array)
			if end - start > 50:
				res.append((start, end))
	return res


def find_digits_positions(img, reserved_threshold=20):
	# cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# digits_positions = []
	# for c in cnts[1]:
	#     (x, y, w, h) = cv2.boundingRect(c)
	#     cv2.rectangle(img, (x, y), (x + w, y + h), (128, 0, 0), 2)
	#     cv2.imshow('test', img)
	#     cv2.waitKey(0)
	#     cv2.destroyWindow('test')
	#     if w >= reserved_threshold and h >= reserved_threshold:
	#         digit_cnts.append(c)
	# if digit_cnts:
	#     digit_cnts = contours.sort_contours(digit_cnts)[0]

	digits_positions = []
	img_array = np.sum(img, axis=0)
	horizon_position = helper_extract(img_array, threshold=reserved_threshold)
	img_array = np.sum(img, axis=1)
	vertical_position = helper_extract(img_array, threshold=reserved_threshold * 4)
	# make vertical_position has only one element
	if len(vertical_position) > 1:
		vertical_position = [(vertical_position[0][0], vertical_position[len(vertical_position) - 1][1])]
	for h in horizon_position:
		for v in vertical_position:
			digits_positions.append(list(zip(h, v)))
	assert len(digits_positions) > 0, "Failed to find digits's positions"

	return digits_positions


def recognize_digits_area_method(digits_positions, output_img, input_img):
	digits = []
	for c in digits_positions:
		x0, y0 = c[0]
		x1, y1 = c[1]
		roi = input_img[y0:y1, x0:x1]
		h, w = roi.shape
		suppose_W = max(1, int(h / H_W_Ratio))
		# 对1的情况单独识别
		if w < suppose_W / 2:
			x0 = x0 + w - suppose_W
			w = suppose_W
			roi = input_img[y0:y1, x0:x1]
		width = (max(int(w * 0.15), 1) + max(int(h * 0.15), 1)) // 2
		dhc = int(width * 0.8)
		# print('width :', width)
		# print('dhc :', dhc)

		small_delta = int(h / arc_tan_theta) // 4
		# print('small_delta : ', small_delta)
		segments = [
			# # version 1
			# ((w - width, width // 2), (w, (h - dhc) // 2)),
			# ((w - width - small_delta, (h + dhc) // 2), (w - small_delta, h - width // 2)),
			# ((width // 2, h - width), (w - width // 2, h)),
			# ((0, (h + dhc) // 2), (width, h - width // 2)),
			# ((small_delta, width // 2), (small_delta + width, (h - dhc) // 2)),
			# ((small_delta, 0), (w, width)),
			# ((width, (h - dhc) // 2), (w - width, (h + dhc) // 2))

			# # version 2
			((w - width - small_delta, width // 2), (w, (h - dhc) // 2)),
			((w - width - 2 * small_delta, (h + dhc) // 2), (w - small_delta, h - width // 2)),
			((width - small_delta, h - width), (w - width - small_delta, h)),
			((0, (h + dhc) // 2), (width, h - width // 2)),
			((small_delta, width // 2), (small_delta + width, (h - dhc) // 2)),
			((small_delta, 0), (w + small_delta, width)),
			((width - small_delta, (h - dhc) // 2), (w - width - small_delta, (h + dhc) // 2))
		]
		# cv2.rectangle(roi, segments[0][0], segments[0][1], (128, 0, 0), 2)
		# cv2.rectangle(roi, segments[1][0], segments[1][1], (128, 0, 0), 2)
		# cv2.rectangle(roi, segments[2][0], segments[2][1], (128, 0, 0), 2)
		# cv2.rectangle(roi, segments[3][0], segments[3][1], (128, 0, 0), 2)
		# cv2.rectangle(roi, segments[4][0], segments[4][1], (128, 0, 0), 2)
		# cv2.rectangle(roi, segments[5][0], segments[5][1], (128, 0, 0), 2)
		# cv2.rectangle(roi, segments[6][0], segments[6][1], (128, 0, 0), 2)
		# cv2.imshow('i', roi)
		# cv2.waitKey()
		# cv2.destroyWindow('i')
		on = [0] * len(segments)

		for (i, ((xa, ya), (xb, yb))) in enumerate(segments):
			seg_roi = roi[ya:yb, xa:xb]
			# plt.imshow(seg_roi)
			# plt.show()
			total = cv2.countNonZero(seg_roi)
			area = (xb - xa) * (yb - ya) * 0.9
			print(total / float(area))
			if total / float(area) > 0.45:
				on[i] = 1

		# print(on)

		if tuple(on) in DIGITS_LOOKUP.keys():
			digit = DIGITS_LOOKUP[tuple(on)]
		else:
			digit = '*'
		digits.append(digit)
		cv2.rectangle(output_img, (x0, y0), (x1, y1), (0, 128, 0), 2)
		cv2.putText(output_img, str(digit), (x0 - 10, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 0), 2)

	return digits


def recognize_digits_line_method(digits_positions, output_img, input_img):
	digits = []
	for c in digits_positions:
		x0, y0 = c[0]
		x1, y1 = c[1]
		roi = input_img[y0:y1, x0:x1]
		h, w = roi.shape
		suppose_W = max(1, int(h / H_W_Ratio))

		# 消除无关符号干扰
		if x1 - x0 < 25 and cv2.countNonZero(roi) / ((y1 - y0) * (x1 - x0)) < 0.2:
			continue

		# 对1的情况单独识别
		if w < suppose_W / 2:
			x0 = max(x0 + w - suppose_W, 0)
			roi = input_img[y0:y1, x0:x1]
			w = roi.shape[1]

		center_y = h // 2
		quater_y_1 = h // 4
		quater_y_3 = quater_y_1 * 3
		center_x = w // 2
		line_width = 5  # line's width
		width = (max(int(w * 0.15), 1) + max(int(h * 0.15), 1)) // 2
		small_delta = int(h / arc_tan_theta) // 4
		segments = [
			((w - 2 * width, quater_y_1 - line_width), (w, quater_y_1 + line_width)),
			((w - 2 * width, quater_y_3 - line_width), (w, quater_y_3 + line_width)),
			((center_x - line_width - small_delta, h - 2 * width), (center_x - small_delta + line_width, h)),
			((0, quater_y_3 - line_width), (2 * width, quater_y_3 + line_width)),
			((0, quater_y_1 - line_width), (2 * width, quater_y_1 + line_width)),
			((center_x - line_width, 0), (center_x + line_width, 2 * width)),
			((center_x - line_width, center_y - line_width), (center_x + line_width, center_y + line_width)),
		]
		on = [0] * len(segments)

		for (i, ((xa, ya), (xb, yb))) in enumerate(segments):
			seg_roi = roi[ya:yb, xa:xb]
			# plt.imshow(seg_roi, 'gray')
			# plt.show()
			total = cv2.countNonZero(seg_roi)
			area = (xb - xa) * (yb - ya) * 0.9
			# print('prob: ', total / float(area))
			if total / float(area) > 0.25:
				on[i] = 1
		# print('encode: ', on)
		if tuple(on) in DIGITS_LOOKUP.keys():
			digit = DIGITS_LOOKUP[tuple(on)]
		else:
			digit = '*'

		digits.append(digit)

		# 小数点的识别
		# print('dot signal: ',cv2.countNonZero(roi[h - int(3 * width / 4):h, w - int(3 * width / 4):w]) / (9 / 16 * width * width))
		if cv2.countNonZero(roi[h - int(3 * width / 4):h, w - int(3 * width / 4):w]) / (9. / 16 * width * width) > 0.65:
			digits.append('.')
			cv2.rectangle(output_img,
			              (x0 + w - int(3 * width / 4), y0 + h - int(3 * width / 4)),
			              (x1, y1), (0, 128, 0), 2)
			cv2.putText(output_img, 'dot',
			            (x0 + w - int(3 * width / 4), y0 + h - int(3 * width / 4) - 10),
			            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 0), 2)

		cv2.rectangle(output_img, (x0, y0), (x1, y1), (0, 128, 0), 2)
		cv2.putText(output_img, str(digit), (x0 + 3, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 0), 2)
	return digits


def main():
	# args = parser.parse_args()
	blurred, gray_img = load_image('Out1.bmp', show=False)
	output = blurred
	dst = preprocess(blurred, THRESHOLD, show=False)
	digits_positions = find_digits_positions(dst)
	digits = recognize_digits_line_method(digits_positions, output, dst)
	# cv2.imshow('output', output)
	# cv2.waitKey()
	# cv2.destroyAllWindows()
	print(digits)


if __name__ == '__main__':
	main()

'''
def cnvt_edged_image(img_arr, should_save=False):
  # ratio = img_arr.shape[0] / 300.0
  image = imutils.resize(img_arr,height=300)
  gray_image = cv2.bilateralFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),11, 17, 17)
  edged_image = cv2.Canny(gray_image, 30, 200)

  if should_save:
    cv2.imwrite('cntr_ocr.jpg')

  return edged_image

def find_display_contour(edge_img_arr):
  display_contour = None
  edge_copy = edge_img_arr.copy()
  contours,hierarchy = cv2.findContours(edge_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  top_cntrs = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

  for cntr in top_cntrs:
    peri = cv2.arcLength(cntr,True)
    approx = cv2.approxPolyDP(cntr, 0.02 * peri, True)

    if len(approx) == 4:
      display_contour = approx
      break

  return display_contour

def crop_display(image_arr):
  edge_image = cnvt_edged_image(image_arr)
  display_contour = find_display_contour(edge_image)
  cntr_pts = display_contour.reshape(4,2)
  return cntr_pts


def normalize_contrs(img,cntr_pts):
  ratio = img.shape[0] / 300.0
  norm_pts = np.zeros((4,2), dtype="float32")

  s = cntr_pts.sum(axis=1)
  norm_pts[0] = cntr_pts[np.argmin(s)]
  norm_pts[2] = cntr_pts[np.argmax(s)]

  d = np.diff(cntr_pts,axis=1)
  norm_pts[1] = cntr_pts[np.argmin(d)]
  norm_pts[3] = cntr_pts[np.argmax(d)]

  norm_pts *= ratio

  (top_left, top_right, bottom_right, bottom_left) = norm_pts

  width1 = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
  width2 = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
  height1 = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
  height2 = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

  max_width = max(int(width1), int(width2))
  max_height = max(int(height1), int(height2))

  dst = np.array([[0,0], [max_width -1, 0],[max_width -1, max_height -1],[0, max_height-1]], dtype="float32")
  persp_matrix = cv2.getPerspectiveTransform(norm_pts,dst)
  return cv2.warpPerspective(img,persp_matrix,(max_width,max_height))

def process_image(orig_image_arr):
  ratio = orig_image_arr.shape[0] / 300.0

  display_image_arr = normalize_contrs(orig_image_arr,crop_display(orig_image_arr))
  #display image is now segmented.
  gry_disp_arr = cv2.cvtColor(display_image_arr, cv2.COLOR_BGR2GRAY)
  gry_disp_arr = exposure.rescale_intensity(gry_disp_arr, out_range= (0,255))

  #thresholding
  ret, thresh = cv2.threshold(gry_disp_arr,127,255,cv2.THRESH_BINARY)
  return thresh

def ocr_image(orig_image_arr):
  otsu_thresh_image = PIL.Image.fromarray(process_image(orig_image_arr))
  return image_to_string(otsu_thresh_image, lang="letsgodigital", config="-psm 100 -c tessedit_char_whitelist=.0123456789")


#print(ocr_image(np.array(NewIm.resize((32,32))).reshape((3,32,32))))

print(pytesseract.image_to_string(Image.open('test.jpg'), lang="letsgodigital", boxes=False, config="digits"))
'''
