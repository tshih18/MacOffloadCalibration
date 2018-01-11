# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import copy
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import multiprocessing
import time

# for transparent objects saturation-10 sharpness-4 else use sharpness-2

# python2 mul_proc_offsetv4.py --image1 images/color_ref1.png --image2 images/color_cal1.png --refwidth 17.9 --sdist 12 --jump 12

# python2 mul_proc_offsetv4.py --image1 images/image_refv2.png --image2 images/image_calv2.png --refwidth 17.9 --sdist 12 --jump 12

# python2 mul_proc_offsetv4.py --image1 images/image_refv1.png --image2 images/image_calv1.png --refwidth 17.9 --sdist 12 --jump 12

# python2 mul_proc_offsetv4.py --image1 images/image_ref2.png --image2 images/image_cal3.png --refwidth 17.9 --sdist 12 --jump 12

# python2 mul_proc_offsetv4.py --image1 images/image_ref2.png --image2 images/image_cal2.png --refwidth 17.9 --sdist 12 --jump 12

# python2 mul_proc_offsetv4.py --image1 images/image_ref1.png --image2 images/image_cal1.png --refwidth 17.9 --sdist 12 --jump 12

# python2 mul_proc_offsetv4.py --image1 images/beta_test82.png --image2 images/beta_test81.png --refwidth 17.9 --sdist 10 --jump 10


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image1", required=True,
	help="path to the input image")
ap.add_argument("-ii", "--image2", required=True,
	help="path to the input image")
ap.add_argument("-w", "--refwidth", type=float, required=True,
	help="width of the reference left-most object in the image (in mm)")
ap.add_argument("-s", "--sdist", type=float, required=True,
	help="starting distance from center of rej obj to next line (in mm)")
ap.add_argument("-j", "--jump", type=float, required=True,
	help="increment  (in mm)")
args = vars(ap.parse_args())

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def process_image((image1, sat, sharp, bright, contra)): # default is original image
##	temp_image = copy.deepcopy(image)

	# print("testing " + str(sat) + " " + str(sharp))

	image = Image.open(image1).convert('RGB')
	width, height = image.size   # Get dimensions
	## these image dimsension are based off points -- I can get these points from edge detection used in offsets
	temp_image = image.crop((0, int(height/5.0), width, height)) # (left, top, right, bottom) (0, height/4.0, width, height)

	saturation = ImageEnhance.Color(temp_image)#victor test
	temp_image = saturation.enhance(sat)#victor test

	sharpness = ImageEnhance.Sharpness(temp_image)#victor test
	temp_image = sharpness.enhance(sharp)#victor test

	brightness = ImageEnhance.Brightness(temp_image)
	temp_image = brightness.enhance(bright)

	contrast = ImageEnhance.Contrast(temp_image)
	temp_image = contrast.enhance(contra)

	image = np.array(temp_image) # opens image in RGB
	image = image[:, :, ::-1].copy() # inverse to BGR for opencv format

	# rotate image here 180 degrees if need be
	# grab the dimensions of the image and calculate the center
	# of the image
	(h, w) = image.shape[:2]
	center = (w / 2, h / 2)

	# rotate the image by 180 degrees
	M = cv2.getRotationMatrix2D(center, 180, 1.0)
	image = cv2.warpAffine(image, M, (w, h))

	image = cv2.fastNlMeansDenoisingColored(image,None,2,10,7,21) #victor test

	img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	gray_im = cv2.fastNlMeansDenoising(img, 2, 31, 7) # denosises grey image to get smoother gradients over gray image

	kernel = None
	kernel = np.ones((3,3), np.uint8) # if needed for erosion or dilation
	# dilate = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
	# erode = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))

	thresh1 = cv2.adaptiveThreshold(gray_im,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
				cv2.THRESH_BINARY,11,2)
	thresh2 = cv2.adaptiveThreshold(gray_im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
				cv2.THRESH_BINARY,11,2)

	thresh = cv2.addWeighted(thresh1,.5,thresh2,.5, 0)

	edge = auto_canny(thresh)
	edge = cv2.dilate(edge, kernel, iterations=1)
	edge = cv2.erode(edge, kernel, iterations=1)

	# print("finished " + str(sat) + " " + str(sharp))

	return edge

def main():
	image = Image.open(args["image1"]).convert('RGB')
	width, height = image.size   # Get dimensions
	## these image dimsension are based off points -- I can get these points from edge detection used in offsets
	image = image.crop((0, int(height/5.0), width, height)) # (left, top, right, bottom) (0, height/4.0, width, height)

	### mul processes start
	data = ([args["image1"], 10, 6, .5, 1], [args["image1"], 1, 1, 1, 1], [args["image1"], 10, 1, 1, 1])

	p = multiprocessing.Pool(4) # assuming quad core
	edge1, edge2, edge3 = p.map(process_image, data)

	# print("If last print then did wait")

	### mul processes end

	# convert image to cv2 style
	image = np.array(image) # opens image in RGB
	image = image[:, :, ::-1].copy() # inverse to BGR for opencv format

	# rotate image here 180 degrees if need be
	# grab the dimensions of the image and calculate the center
	# of the image
	(h, w) = image.shape[:2]
	center = (w / 2, h / 2)

	# rotate the image by 180 degrees
	M = cv2.getRotationMatrix2D(center, 180, 1.0)
	image = cv2.warpAffine(image, M, (w, h))

	# #########################################################

	final_edged = cv2.addWeighted(edge1,.5,edge2,.5,0)

	ret,final_edged = cv2.threshold(final_edged,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	final_edged = cv2.addWeighted(final_edged,.5,edge3,.5,0)

	ret,final_edged = cv2.threshold(final_edged,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# find contours in the edge map
	cnts = cv2.findContours(final_edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	# sort the contours from left-to-right and initialize the
	# 'pixels per metric' calibration variable
	(cnts, _) = contours.sort_contours(cnts)
	pixelsPerMetric = None

	#victor -- attempt to get 2 largest areas

	reference_obj = None
	filpos = 0
	while reference_obj is None:
		if cv2.contourArea(cnts[filpos]) > 1000: #counters possible small leftmost objects not the square
			reference_obj = copy.deepcopy(cnts[filpos])
			cnts = cnts[:filpos] + cnts[filpos+1:]
			filament = False
		else :
			filpos += 1


	index = 0
	indices = list()
	contour_areas = list()
	for c in cnts:

		if (cv2.contourArea(c) > 1000):
			indices.append(index)
			contour_areas.append(cv2.contourArea(c))

		index += 1

	sorted_cnts_by_areas = [cnts[x] for y, x in zip(contour_areas, indices)]
	sorted_cnts_by_areas.insert(0, reference_obj)

	sorted_cnts_by_areas = sorted_cnts_by_areas[:2] # reference obj and leftmost obj

	leftmost_obj_width = 0

	# loop over the contours individually
	for c in sorted_cnts_by_areas:

		# compute the rotated bounding box of the contour
		orig = image.copy()
		c = cv2.convexHull(c) #victor added
		box = cv2.minAreaRect(c)

		# The output of cv2.minAreaRect() is ((x, y), (w, h), angle). Using cv2.cv.BoxPoints() is meant to convert this to points.
		# ( center (x,y), (width, height), angle of rotation )


	############# Experimental
		(x, y), (w, h), angle = box

		# angle = 0 #converts angle to 0 rotation so its no longer the minimum bounding box

		if (90 - abs(angle) < abs(angle) - 0):
			angle = 90
		else :
			angle = 0

		box = ((x, y), (w, h), angle)
	#############

		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")

		# order the points in the contour such that they appear
		# in top-left, top-right, bottom-right, and bottom-left
		# order, then draw the outline of the rotated bounding
		# box
		box = perspective.order_points(box)

		# unpack the ordered bounding box, then compute the midpoint
		# between the top-left and top-right coordinates, followed by
		# the midpoint between bottom-left and bottom-right coordinates
		(tl, tr, br, bl) = box
		(tltrX, tltrY) = midpoint(tl, tr)
		(blbrX, blbrY) = midpoint(bl, br)

		# compute the midpoint between the top-left and top-right points,
		# followed by the midpoint between the top-righ and bottom-right
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)

		# compute the Euclidean distance between the midpoints
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY)) # width


		# if the pixels per metric has not been initialized, then
		# compute it as the ratio of pixels to supplied metric
		# (in this case, mm)
		if pixelsPerMetric is None:
			pixelsPerMetric = dB / args["refwidth"]

		# compute the size of the object
		leftmost_obj_width = dB / pixelsPerMetric #dimB is width of leftmost obj excluding reference obj


	# print("leftmost obj exlcuding ref obj width in mm: " + str(leftmost_obj_width))


	###############################################################################################################################


	image = Image.open(args["image2"]).convert('RGB')
	width, height = image.size   # Get dimensions
	## these image dimsension are based off points -- I can get these points from edge detection used in offsets
	image = image.crop((0, int(height/5.0), width, height)) # (left, top, right, bottom) (0, height/4.0, width, height)

	### mul processes start
	data = ([args["image2"], 10, 6, .5, 1], [args["image2"], 1, 1, 1, 1], [args["image2"], 10, 1, 1, 1])

	#dont need to reinstantiate pool will cuase error
	edge1, edge2, edge3 = p.map(process_image, data)

	# print("If last print then did wait")

	### mul processes end

	# convert image to cv2 style
	image = np.array(image) # opens image in RGB
	image = image[:, :, ::-1].copy() # inverse to BGR for opencv format

	# rotate image here 180 degrees if need be
	# grab the dimensions of the image and calculate the center
	# of the image
	(h, w) = image.shape[:2]
	center = (w / 2, h / 2)

	# rotate the image by 180 degrees
	M = cv2.getRotationMatrix2D(center, 180, 1.0)
	image = cv2.warpAffine(image, M, (w, h))

	# #########################################################

	final_edged = cv2.addWeighted(edge1,.5,edge2,.5,0)

	ret,final_edged = cv2.threshold(final_edged,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	final_edged = cv2.addWeighted(final_edged,.5,edge3,.5,0)
	# find contours in the edge map
	cnts = cv2.findContours(final_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	# sort the contours from left-to-right and, then initialize the
	# distance colors and reference object
	(cnts, _) = contours.sort_contours(cnts)
	colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),
		(255, 0, 255))
	refObj = None

	reference_obj = None
	filpos = 0
	while reference_obj is None:
		if cv2.contourArea(cnts[filpos]) > 1000: #counters possible small leftmost objects not the square
			reference_obj = copy.deepcopy(cnts[filpos])
			cnts = cnts[:filpos] + cnts[filpos+1:]
			filament = False
		else :
			filpos += 1


	index = 0
	indices = list()
	contour_areas = list()
	for c in cnts:

		if (cv2.contourArea(c) > 1000):
			indices.append(index)
			contour_areas.append(cv2.contourArea(c))

		index += 1

	sorted_cnts_by_areas = [cnts[x] for y, x in zip(contour_areas, indices)]
	sorted_cnts_by_areas.insert(0, reference_obj)

	# adjustments needed for offset
	exactdist = args["sdist"]
	adjustedwidth = 0
	goodprint = True
	offset_changes = [True]

	# loop over the contours individually
	for c in sorted_cnts_by_areas:

		# print("contour area: " +  str(cv2.contourArea(c)))

		# compute the rotated bounding box of the contour
		c = cv2.convexHull(c) #victor added
		box = cv2.minAreaRect(c)

		# The output of cv2.minAreaRect() is ((x, y), (w, h), angle). Using cv2.cv.BoxPoints() is meant to convert this to points.
		# ( center (x,y), (width, height), angle of rotation )


	############# Experimental --- unsure if it actually works
		(x, y), (w, h), angle = box

		# angle = 0 #converts angle to 0 rotation so its no longer the minimum bounding box

		if (90 - abs(angle) < abs(angle) - 0):
			angle = 90
		else :
			angle = 0

		box = ((x, y), (w, h), angle)
	#############

		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int") # makes the points integers instead of floats

		# order the points in the contour such that they appear
		# in top-left, top-right, bottom-right, and bottom-left
		# order, then draw the outline of the rotated bounding
		# box
		box = perspective.order_points(box)

		(tl, tr, br, bl) = box
		(cX, cY) = midpoint(tl, tr) #midpoint of tl and tr

		# compute the midpoint between the top-left and top-right points,
		# followed by the midpoint between the top-righ and bottom-right
		(tltrX, tltrY) = midpoint(tl, tr)
		(blbrX, blbrY) = midpoint(bl, br)
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)

		# if this is the first contour we are examining (i.e.,
		# the left-most contour), we presume this is the
		# reference object
		# this code computes the distance between centers of the bounding box and draws lines
		if refObj is None:
			# unpack the ordered bounding box, then compute the
			# midpoint between the top-left and top-right points,
			# followed by the midpoint between the top-right and
			# bottom-right

			# print("\nmidpoint of ref obj: " + str((cX, cY)))

			(tlblX, tlblY) = midpoint(tl, bl)
			(trbrX, trbrY) = midpoint(tr, br)

			# compute the Euclidean distance between the midpoints,
			# then construct the reference object
			# dist.euclidean((tlblX, tlblY), (trbrX, trbrY)) this is always the pixel width compared against the designated width
			D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
			refObj = (box, (cX, cY), D / leftmost_obj_width)

			## refObj[2] represents the number of pixels per args["width"] in mm
			adjustedwidth = refObj[2]

			# compute the Euclidean distance between the midpoints
			dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY)) / refObj[2]
			dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY)) / refObj[2]

			continue

		# print("midpoint of obj: " + str((cX, cY)))

		# draw the contours on the image
		orig = image.copy()

		# compute the Euclidean distance between the midpoints
		dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY)) / refObj[2]
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY)) / refObj[2]

		## refObj[2] represents the number of pixels per args["width"] in mm

		# stack the reference coordinates and the object coordinates
		# to include the object center
		refCoords = np.vstack([refObj[1]])
		objCoords = np.vstack([(cX, cY)])

		# loop over the original points
		for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
			# draw circles corresponding to the current points and
			# connect them with a line

			calcd_pixeldist = adjustedwidth * exactdist
			actual_pos = (xA + calcd_pixeldist , yA)
			temp_pos = (xB, actual_pos[1])

			# print("\nactual wanted x position: " + str(actual_pos[0]))
			# print("actual wanted y position: " + str(actual_pos[1]))
			# print("actual printed x pos: " + str(xB))
			# print("actual printed y pos: " + str(yB) + "\n")

			if xB > actual_pos[0]:
				# print("x euc difference: " + str(-1 * dist.euclidean(temp_pos, actual_pos)) + " pixels")
				xdiff = -1 * dist.euclidean(temp_pos, actual_pos) / adjustedwidth

			else:
				# print("x euc difference: " + str(dist.euclidean(temp_pos, actual_pos)) + " pixels")
				xdiff = dist.euclidean(temp_pos, actual_pos) / adjustedwidth

			if yB < actual_pos[1]:
				# print("y euc difference " + str(dist.euclidean(temp_pos, (xB, yB))) + " pixels")
				ydiff = dist.euclidean(temp_pos, (xB, yB)) / adjustedwidth
			else:
				# print("y euc difference " + str(-1 * dist.euclidean(temp_pos, (xB, yB))) + " pixels")
				ydiff = -1 * dist.euclidean(temp_pos, (xB, yB)) / adjustedwidth

			if (abs(xdiff) > 20 or abs(ydiff) >20): # distance to change is larger than usual flag as failure to inspect
				goodprint = False
				break

			# print("\nnegative is left/up, positive is right/down")
			# print("x actual diff (printed pos to actual wanted pos): " + str(xdiff) + " mm")
			# print("y actual diff (printed pos to actual wanted pos): " + str(ydiff) + " mm\n")

			# print("estimated distance from ref obj center w/ adjusted coords: "  + str(dist.euclidean((xA, yA), (xB + (xdiff* adjustedwidth) , yB + (ydiff* adjustedwidth))) / refObj[2]) + " mm")
			# print("goal distance to achieve: " + str(exactdist) + " mm\n")

			# generate actual offsets for marlin ---victor
			if (xdiff < 0): # positive left
				if (ydiff < 0): # positive up
					offset_changes.append((abs(xdiff), abs(ydiff)))
				else: # ydiff > 0 and down is negative
					offset_changes.append((abs(xdiff), -1* ydiff))
			elif (xdiff > 0): # moving right is negative
				if (ydiff < 0): # positive up
					offset_changes.append((-1* xdiff, abs(ydiff)))
				else: # ydiff > 0 and down is negative
					offset_changes.append((-1* xdiff, -1* ydiff))


			exactdist += args["jump"]
			# compute the Euclidean distance between the coordinatess		# and then convert the distance in pixels to distance in
			# units
			# VICTORS MATH WOULD GO HERE ROUGHLY
			# print("\neuclidean pixel distance from current midpoint to midpoint: "  + str(dist.euclidean((xA, yA), (xB, yB))))
			# print("adjusted width: "  + str(refObj[2]) + " pixels per mm")
			D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
			(mX, mY) = midpoint((xA, yA), (xB, yB))

			# print("current measured distance from centers: " + str(D) + "\n")

	# print("number of detected objs: " + str(len(offset_changes)))

	if (goodprint):
		print(str(offset_changes))
	else:
		print([False, False])

	''' To be used for esaacs code
	import ast
	offset_list = subprocess.check_output(["python", "offset_algorithm.py --image1 images/beta_test82.png --image2 images/beta_test81.png --refwidth 17.9 --sdist 10 --jump 10"])
	offset_list = ast.literal_eval(offset_list)
	'''

if __name__=="__main__":
	multiprocessing.freeze_support()
	main()
