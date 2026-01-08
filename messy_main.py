"""messy attempt at combining detect_blinks & interactive alphabet board. Working, but low readability + reusability"""

# USAGE
# python messy_main.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
# add on: also import matplotlib to visualise "blink signal"
import matplotlib.pyplot as plt

# create time series for logging blink events
timeseries,totalCount,altSignal, blinkLen = ([],[],[],[])
framenumber = 0
# also create time series for logging

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold

# TODO : add calibration function for EYE_AR_THRESH at beginning of script
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 8
MORSE_PAUSE = 8

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
# for morse pause, my addition:
PAUSE_COUNTER = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
# use video file if provided via args, else use webcam
if args["video"]:
	vs = FileVideoStream(args["video"]).start()
	fileStream = True
else:
	### use VideoStream to enable webcam
	vs = VideoStream(src=0).start()
	fileStream = False

time.sleep(1.0)

########
import pygame
 
# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
GREENISH = (220, 255, 220)
RED = (255, 0, 0)

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 60
HEIGHT = 40
 
# This sets the margin between each cell
MARGIN = 5

# Create a 2 dimensional array. A two dimensional
# array is simply a list of lists.
grid = []
# use gridsize if square:
ROWS = 7
COLS = 7
for row in range(ROWS):
	# Add an empty array that will hold each cell
	# in this row
	grid.append([])
	for column in range(COLS):
		grid[row].append(0)  # Append a cell
 
# alphabet_string = "abcdefghijklmnopqrstuvwxyz"
# link to CIR board: 
# http://www.instructables.com/id/Communication-Board-for-Individuals-with-Disabilit/
CIRORDER = "aeiou5_bfjpv6.cgkqw7?dhlrx8!13msy9$24ntz0@"
# pad it with one extra column worth of spaces
alphabet = " "*ROWS + CIRORDER
# create alphabet position dict that we will populate later in grid loop
alphabet_pos = {}
# MESSAGE string to log to console in course of game
MESSAGE = ""

# Set row 1, cell 5 to one. (Remember rows and
# column numbers start at zero.)
# grid[1][5] = 1

# Initialize pygame
pygame.init()

# Set the HEIGHT and WIDTH of the screen
winHeight = ROWS*(HEIGHT+MARGIN)+ MARGIN
winWidth = COLS*(WIDTH+MARGIN) + MARGIN
WINDOW_SIZE = [winWidth, winHeight]
screen = pygame.display.set_mode(WINDOW_SIZE)
 
# Set title of screen
pygame.display.set_caption("Array Backed Grid")
 
# Loop until the user clicks the close button.
done = False
 
# Used to manage how fast the screen updates
clock = pygame.time.Clock()
 
# render some letters
font = pygame.font.SysFont('arial', 40)

# initialise first tile for beginning of game:
currentTile = (0,0)
# toggle_grid(*currentTile)
# also, in beginning we want the cursor to move downwards 
downwards = True
#######
# get text objects:
def text_objects(text):
	"""takes string, returns a textSurface object and its rectangular coords"""
	textSurface = font.render(text, True, (0, 50, 0))
	return textSurface, textSurface.get_rect()

# create own event for timer interruption, blink
cursor_timer = pygame.USEREVENT + 1
blinkRef = pygame.USEREVENT + 2
blinkEvent = pygame.event.Event(blinkRef)
CURSORDELAY = 800
pygame.time.set_timer(cursor_timer, CURSORDELAY)

# -------- Event handlers  -----------
def clickToggle():
	"""toggles value of tile that is clicked to change color"""
	# User clicks the mouse. Get the position
	pos = pygame.mouse.get_pos()
	# Change the x/y screen coordinates to grid coordinates
	column = pos[0] // (WIDTH + MARGIN)
	row = pos[1] // (HEIGHT + MARGIN)
	# toggle value of that tile
	toggle_grid(row, column)
	# print("Click ", pos, "Grid coordinates: ", row, column)
	clicked_str = alphabet_pos[(row, column)]
	console_msg = "Clicked char {0} at grid point {1}:{2}".format(clicked_str,row, column)
	# could add functionality to log selected chars to message string
	# for now just print one selected char at a time
	print (console_msg) 


def toggle_grid(row, column, only_one_active=True):
	# """toggle a single grid position's value between 0 and 1"""
	if grid[row][column] == 0:
		grid[row][column] = 1
	else:
		grid[row][column] = 0

def inc_before_end(c,end):
	"""mini function to increment counter until end of list then reset"""
	if c < end -1:
		return c+1
	else: 
		return 0


def movingCursor(currentTile, downwards):
	"""move cursor down or sideways based on timer"""
	x, y = currentTile
	if downwards:
		# move down as long until active hits lower end, then reset
		x = inc_before_end(x, ROWS)
	else:
		# move sideways, i.e. increment col instead
		y = inc_before_end(currentTile[1], COLS)
	# return tuple with new position
	return (x,y)

########
# loop over frames from the video stream
while not done:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		#if limit > 1:
		#	print "more than one rect"
		#	break
		# limit +=1
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# my addition - do this anyway
		altSignal.append(0)
		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1
			timeseries.append((framenumber, 1))
			# dont want cursor to move forward during blink, makes timing annoying!
			pygame.event.set_blocked(cursor_timer)
			# also, since eyes aren't open anymore
			# PAUSE_COUNTER = 0
		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			# if the eyes were OPEN for sufficient number then add morse pause
			# PAUSE_COUNTER += 1
			# if PAUSE_COUNTER >= MORSE_PAUSE:
			#	blinkLen.append(-1)
			#	PAUSE_COUNTER = 0 

			# if the eyes were CLOSED for a sufficient number of
			# then increment the total number of blinks
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
				# my addition: if long enough for blink to occur,
				# retroactively set those "blink frames" to one in altSignal
				startBlinkFrame = framenumber - COUNTER
				theBlink = [1 for i in altSignal[startBlinkFrame:framenumber]]
				altSignal[startBlinkFrame:framenumber] = theBlink
				blinkLen.append(len(theBlink))
				# message for logging:
				# print "reassigned:" , altSignal[startBlinkFrame:framenumber]
				# also post pygame blink event, and allow cursor to move again
				pygame.event.set_allowed(cursor_timer)
				pygame.event.post(blinkEvent)
			timeseries.append((framenumber,0))
			# TODO: call morse function on list so far
			# if char detected, send new cv2.putText to section below
			# 

			# reset the eye frame counter
			COUNTER = 0

		# draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.putText(frame, "Eye Aspect Ratio: {:.2f}".format(ear), (250, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	totalCount.append((framenumber, TOTAL))
	framenumber += 1
	### event handling part...
	for event in pygame.event.get():  # User did something
		if event.type == pygame.QUIT:  # If user clicked close
			done = True  # Flag that we are done so we exit this loop
		elif event.type in (pygame.MOUSEBUTTONDOWN, blinkRef):
			if currentTile[1] != 0:
				# if we are on one of the letter rows
				MESSAGE = MESSAGE + alphabet_pos[currentTile]
				print (MESSAGE)
				currentTile = (0,0)
				downwards = True
			elif currentTile[1] == 0 and not downwards:
				# if we are moving sideways in the empty row and want to reset
				currentTile = (0,0)
				downwards = True
			elif downwards:
				# if we are in empty row and want to start moving sideways
				downwards = False
		elif event.type == cursor_timer:
			# cycle cursor through first row
			currentTile = movingCursor(currentTile, downwards)
			# toggle_grid(*currentTile)


	### rendering part...
	# Set the screen background
	screen.fill(BLACK)
	# Draw the grid
	# extra string_counter for CIR board implementation
	string_counter = 0
	for column in range(COLS):
		for row in range(ROWS):
			color = WHITE
			# old: if grid[row][column] == 1:
			# new: recolor currently active row in lighter color
			if row == currentTile[0]:
				color = GREENISH
			# also highlight active tile in strong color:
			if (row, column) == currentTile:
				color = GREEN
			x_coord = (MARGIN + WIDTH) * column + MARGIN
			y_coord = (MARGIN + HEIGHT) * row + MARGIN
			pygame.draw.rect(screen,
							 color,
							 [x_coord, y_coord, WIDTH, HEIGHT])
			# in loop also insert CIR board chars on screen:
			currentChar = alphabet[string_counter]
			surf, rect = text_objects(currentChar)
			screen.blit(surf, (x_coord, y_coord))
			# index chars in dict with row/column tuple as key:
			alphabet_pos.update({(row, column): currentChar})
			string_counter += 1
	# Limit to 60 frames per second
	clock.tick(60)
	# timer test code:
	# milliseconds += clock.tick_busy_loop(60)
	# Go ahead and update the screen with what we've drawn.
	pygame.display.flip()
 	# increment framenumber counter

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break


# Be IDLE friendly. If you forget this line, the program will 'hang'
# on exit.
pygame.quit()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# now display blink timeseries as step plot:
def timeAxis(series):
	# returns incrementing list of length of the recorded series
	return [t for t in range(len(series))]

def plotWrangler(series):
	# returns t and values as tuple for matplotlib conveninence
	# so I only have to change one var at a time for testing
	t = timeAxis(series)
	val = series
	return (t, val)

#plt.xlim(0, 100)
#plt.ylim(0, TOTAL)
#plt.step(x, timeseries)
print (blinkLen)
x, y = plotWrangler(altSignal)
#plt.plot(timeAxis(altSignal), altSignal)
plt.step(x, y)
plt.show()



