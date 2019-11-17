import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as pt
from matplotlib.lines import Line2D

# File Selection using tkinter
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

image = cv2.imread(file_path)
original = cv2.imread(file_path)

# Image Colour Conversion -> Get Threshold
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)[1]

horizontal = np.copy(thresh)
vertical = np.copy(thresh)

# Variable in order to increase/decrease the amount of lines to be detected
scale = 15

# Get number of columns
horizontalCols = horizontal.shape[1]

# Size for structuring element(SE)
horizontalSize = horizontalCols / scale

# Get SE for searching for lines process
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (int(horizontalSize), 1))

# Perform erosion & dilation
#horizontal = cv2.erode(horizontal, horizontalStructure)
#horizontal = cv2.dilate(horizontal, horizontalStructure)
horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontalStructure)

# Repeat the same process for vertical from horizontal
verticalRows = vertical.shape[0]
verticalSize = verticalRows / scale
verticalSize = int(verticalSize)

verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalSize))

#vertical = cv2.erode(vertical, verticalStructure)
#vertical = cv2.dilate(vertical, verticalStructure)
vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, verticalStructure)

# create a mask that combine both horizontal & vertical
# to make up the table in the image
mask = horizontal + vertical

# find the contours from mask
mask_cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# loop through the contours to complete the mask
for c in mask_cnts:
    x, y, w, h = cv2.boundingRect(c)
    mask[y:y + h + 1, x:x + w + 1] = 255

# inverted mask to remove tables from image
mask_noTable = cv2.bitwise_not(mask)

# to extract table
threshold_justTable = cv2.bitwise_and(thresh, mask)

# to extract texts
threshold_noTable = cv2.bitwise_and(thresh, mask_noTable)

# additional SE for dilation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# dilation process for both noTable and justTable
threshold_noTable_dilate = cv2.dilate(threshold_noTable, kernel, iterations=4)
threshold_justTable_dilate = cv2.dilate(threshold_justTable, kernel, iterations=4)

# text contours, to highlight text paragraphs
# in green border line
text_ctns = cv2.findContours(threshold_noTable_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
for c in text_ctns:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# table only contours, to highlight tables
# in red border line
table_ctns = cv2.findContours(threshold_justTable_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
for c in table_ctns:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

custom_lines = [Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='r', lw=2)]

# Plot the original document and analysed document
pt.figure()
pt.interactive(False)
pt.suptitle(file_path)
pt.subplot(1, 2, 1)
pt.title("Original Document")
pt.imshow(original)
pt.subplot(1, 2, 2)
pt.title('Analysed Document')
pt.imshow(image)
pt.legend(custom_lines, ['Text', 'Table'])
pt.show(block=True)










