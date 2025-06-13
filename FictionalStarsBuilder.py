# Fictional Stars Builder - Yossi Modified
#   Creating Custom Constellations for the Celestial Vault Plugin
# 
# original by Alban BERGERET
# edited by Yossi40100 06/12/2025

# Import required Libraries
import os
import sys
import csv
import math
import numpy as np

import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox as mb

import PIL
from PIL import Image, ImageTk, ImageOps

import cv2 as cv

import image2starpoint

# Global Variables
image_width = 1920
image_height = 1080
input_fileName = "./yossi40100logo.png"
input_image = None
input_imageDot = None
input_image_grayscale = None
stars_blobs = None
min_star_size = 0
max_star_size = 0
data = []

# UI
main_window = tk.Tk()
display_mode = tk.IntVar()
star_num = tk.IntVar()
draw_stars_blobs = tk.IntVar()
threshold_min = tk.DoubleVar()
blob_min_area = tk.DoubleVar()
blob_max_area = tk.DoubleVar()
min_dist_between_blobs = tk.DoubleVar()
min_magnitude = tk.DoubleVar()
max_magnitude = tk.DoubleVar()
RACentor = tk.DoubleVar()
DECCentor= tk.DoubleVar()
RASize = tk.DoubleVar()
DECSize = tk.DoubleVar()
deg = tk.DoubleVar()
status_label = None
image_label = None



def open_picture():
    filename = fd.askopenfilename(filetypes=[("Picture files", ".bmp .jpeg .jpg .jpe .jp2 .png .tiff .tif")])
    loaded_image = cv.imread(filename)
    if loaded_image is None:
        mb.showerror("Error loading file", "Maybe check for unicode chars in the file name, they are not supported...")
    else:
        global input_fileName
        global input_image
        global input_image_grayscale
        global input_imageDot
        input_fileName = filename
        input_image = loaded_image
        on_starnum_change(int(star_num.get()))

def on_starnum_change(val):
    global input_fileName
    global input_image_grayscale
    global input_imageDot
    input_imageDot = image2starpoint.generate_star_constellation_bgr(input_fileName,int(val))
    input_image_grayscale = cv.cvtColor(input_imageDot, cv.COLOR_BGR2GRAY)

def compute_final_star():
    global stars_blobs
    global data

    data = []
    data.append(['ID','Name','RA', 'DEC', 'Magnitude', 'Color'])

    rad = np.radians(float(deg.get()))
    rot_mat = np.array([
        np.cos(rad), -np.sin(rad),
        np.sin(rad),  np.cos(rad)
    ])

    id=0
    for current_point in stars_blobs:
        id = id + 1

        # Compute Normalized size
        point_x_pixels, point_y_pixels = current_point.pt
        input_image_width, input_image_height, _ = input_imageDot.shape
        x_normalized = 2 * (point_x_pixels / input_image_width - 0.5)
        y_normalized = -2 * (point_y_pixels / input_image_height - 0.5)  # invert Y 

        # Compute Magnitude
        point_diameter = current_point.size
        xp = np.array([min_star_size, max_star_size])
        fp = np.array([max_magnitude.get(), min_magnitude.get()])
        magnitude = float(np.interp(np.array([point_diameter], dtype='float64'), xp, fp)[0])

        # Compute the Color
        # 1 - Extract ROI
        start_column = round(point_x_pixels - point_diameter / 2)
        end_column = round(point_x_pixels + point_diameter / 2 + 1)
        start_row = round(point_y_pixels - point_diameter / 2)
        end_row = round(point_y_pixels + point_diameter / 2 + 1)
        roi = input_imageDot[start_row:end_row, start_column:end_column]

        # 2 - The ROI is rectangular, so for more accurate computations, we'll use a circular mask, drawing the Blob in white. Shift is for sub-pixel accuracy
        mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
        center_coordinates = (point_x_pixels - start_column, point_y_pixels - start_row)
        radius = point_diameter / 2.0
        shift = 8
        center_coordinates_shifted = (round(center_coordinates[0] * 2 ** shift), round(center_coordinates[1] * 2 ** shift))
        radius_shifted = round(radius * 2 ** shift)
        mask = cv.circle(mask, center_coordinates_shifted, radius_shifted, (255, 255, 255), -1, lineType=cv.LINE_AA, shift=shift)
        color = cv.mean(roi, mask)

        # 3 - Calculate position on the celestial sphere
        ra  = 0.5 * (y_normalized*rot_mat[1] - x_normalized*rot_mat[0]) * float(RASize.get())    + float(RACentor.get())
        dec = 0.5 * (y_normalized*rot_mat[3] - x_normalized*rot_mat[2]) * float(DECSize.get()) + float(DECCentor.get())

        data.append(["%d" % id, "", "%.5f" % ra, "%.5f" % dec, "%.2f" % magnitude, "(R=%.3f,G=%.3f,B=%.3f)" % (color[2] / 255, color[1] / 255, color[0] / 255)])

# Convert RA and DEC coordinates to XY on celestial map
def ra_dec_to_xy(ra, dec, width, height):
    x = int((1 - ra / 24.0) * width) % width
    y = int(((90 - dec) / 180.0) * height) % height
    return x, y

# Draw stars at positions on the celestial expansion
def plot_final():
    global data
    img_width = 800
    img_height = 400

    background = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    keypoints = []
    for star in data[1:]:
        x, y = ra_dec_to_xy(float(star[2]), float(star[3]), img_width, img_height)
        kp = cv.KeyPoint(x=float(x), y=float(y), size=1)
        keypoints.append(kp)

    final_image = cv.drawKeypoints(
        background,
        keypoints,
        None,
        color=(255, 255, 255),
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return final_image


def export_csv():
    global data
    compute_final_star()
    output_filename = os.path.splitext(input_fileName)[0] + ".csv"
    with open(output_filename, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile)
        writer.writerows(data)
    mb.showinfo("Export successful", "%s exported successfully" % output_filename)


def on_image_label_size_changed(event):
    global image_width
    global image_height
    image_width = event.width
    image_height = event.height


# Define function to show frame
def show_frames():
    global stars_blobs
    global min_star_size
    global max_star_size
    global status_label

    if not input_image is None : 
        # Do OpenCV actions
        # Step 1 - Threshold
        ret, cv_image_threshold = cv.threshold(input_image_grayscale, threshold_min.get(), 255.0, cv.THRESH_TOZERO)
        cv_image_threshold_inverted = cv.bitwise_not(cv_image_threshold)
    
        # Step 2 - Extract Blobs
        params = cv.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = max(1.0, blob_min_area.get())
        params.maxArea = max(blob_min_area.get(), blob_max_area.get())
        params.minDistBetweenBlobs = max(min_dist_between_blobs.get(), 0.01)
        params.minThreshold = threshold_min.get()
        params.maxThreshold = 255
        params.thresholdStep = 10
    
        detector = cv.SimpleBlobDetector_create(params)
        stars_blobs = detector.detect(cv_image_threshold_inverted)
    
        # Update Status Bar
        max_star_size = sys.float_info.min
        min_star_size = sys.float_info.max
        found_stars_count = len(stars_blobs)
    
        for CurrentPoint in stars_blobs:
            if CurrentPoint.size > max_star_size:
                max_star_size = CurrentPoint.size
            if CurrentPoint.size < min_star_size:
                min_star_size = CurrentPoint.size
    
        status_text = "Found Stars Count = " + str(found_stars_count)
        if found_stars_count > 0:
            status_text += " | Diameter : Min %.2f (%.2f px²), Max: %.2f (%.2f px²)" % (
                min_star_size, math.pi * min_star_size * min_star_size / 4.0, max_star_size,
                math.pi * max_star_size * max_star_size / 4.0)
        status_label.config(text=status_text)
    
        # Select the input image we'll take to draw blobs DisplayModes = ["Original", "GrayScale", "Threshold"]
        background_image = None
        if display_mode.get() == 4:
            compute_final_star()
            final_image = plot_final()
        else:
            if display_mode.get() == 0:
                background_image = input_image
            elif display_mode.get() == 1:
                background_image = input_imageDot
            elif display_mode.get() == 2:
                background_image = input_image_grayscale
            elif display_mode.get() == 3:
                background_image = cv_image_threshold

        
            # Draw detected blobs as red circles. cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
            if draw_stars_blobs.get() == 1:
                final_image = cv.drawKeypoints(background_image, stars_blobs, np.array([]), (0, 0, 255),
                                            cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            else:
                final_image = background_image
        
        # Now Convert to PIL format - Convert to RGB, then to an Array        
        final_image_rgb = cv.cvtColor(final_image, cv.COLOR_BGR2RGB)
        final_image_array = Image.fromarray(final_image_rgb)
        final_image_array_resized = PIL.ImageOps.contain(final_image_array, (image_width, image_height))
    
        # Convert image to PhotoImage
        finale_image_tk = ImageTk.PhotoImage(final_image_array_resized)
    
        image_label.configure(image=finale_image_tk)
        image_label.imgtk = finale_image_tk

    # Refresh in 20mn
    main_window.after(20, show_frames)


def build_ui():
    global main_window
    global image_label
    global status_label
    # Create an instance of TKinter Window or frame, and 3 Main frames
    main_window.geometry("1920x1080")
    main_window.title("Fictional Stars Builder - Yossi Modified")
    main_window.columnconfigure(0, weight=1)
    main_window.columnconfigure(1, weight=1)
    main_window.columnconfigure(2, weight=0)
    main_window.rowconfigure(0, weight=1)
    main_window.rowconfigure(1, weight=0)

    left_frame = tk.Frame(main_window)
    left_frame.columnconfigure(0, weight=1)
    left_frame.grid(row=0, column=0, sticky=tk.NSEW)

    center_frame = tk.Frame(main_window)
    center_frame.columnconfigure(0, weight=1)
    center_frame.rowconfigure(0, weight=1)
    center_frame.grid(row=0, column=1, sticky=tk.NSEW)

    right_frame = tk.Frame(main_window, pady=5, padx=5)
    right_frame.grid(row=0, column=2, sticky=tk.N)

    # Build Settings UI on LeftFrame
    # -Display
    settings_current_row = 0
    (tk.Label(left_frame, text="Display", pady=5)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.W))

    display_modes_frame = tk.Frame(left_frame)
    display_modes_frame.grid(row=(settings_current_row := settings_current_row + 1), column=0, sticky=tk.W)

    display_modes = ["Original","Dot", "GrayScale", "Threshold", "FinalImage"]
    for index in range(len(display_modes)):
        rb = tk.Radiobutton(display_modes_frame, text=display_modes[index], variable=display_mode, value=index)
        rb.grid(row=0, column=index, sticky=tk.EW, padx=5)

    cb = tk.Checkbutton(display_modes_frame, text="Draw Stars Blobs", variable=draw_stars_blobs)
    cb.grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.W, padx=5)


    # -StarNum
    (tk.Label(left_frame, text="Number of Original Star", pady=5)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.W))
    (tk.Scale(left_frame, from_=0.0, to=500.0, digits=1, orient=tk.HORIZONTAL, variable=star_num,command=on_starnum_change)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.EW, padx=5))


    # -Threshold
    (tk.Label(left_frame, text="Image Threshold Min", pady=5)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.W))
    (tk.Scale(left_frame, from_=0.0, to=255.0, digits=1, orient=tk.HORIZONTAL, variable=threshold_min)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.EW, padx=5))

    # -Blob Min Size
    (tk.Label(left_frame, text="Blob Min Area (pixels²)", pady=5)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.W))
    (tk.Scale(left_frame, from_=0, to=100, digits=1, orient=tk.HORIZONTAL, variable=blob_min_area)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.EW))

    # -Blob Max Size
    (tk.Label(left_frame, text="Blob Max Area (pixels²)", pady=5)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.W))
    (tk.Scale(left_frame, from_=0, to=5000, digits=1, orient=tk.HORIZONTAL, variable=blob_max_area)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.EW))

    # -Minimum Distance Between Blobs
    (tk.Label(left_frame, text="Minimum Distance Between Blobs", pady=5)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.W))
    (tk.Scale(left_frame, from_=0.1, to=10, digits=3, resolution=0.1, orient=tk.HORIZONTAL, variable=min_dist_between_blobs, activebackground="SteelBlue2")
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.EW))

    # -Magnitudes 
    (tk.Label(left_frame, text="Magnitudes", pady=5)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.W))
    magnitude_frame = tk.Frame(left_frame)
    magnitude_frame.grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.W)
    tk.Label(magnitude_frame, text="Brightest Star Magnitude", pady=5).grid(row=0, column=0, sticky=tk.W)
    tk.Spinbox(magnitude_frame, from_=-5, to=15, textvariable=min_magnitude).grid(row=0, column=1, sticky=tk.W)
    tk.Label(magnitude_frame, text="Faintest Star Magnitude", pady=5).grid(row=0, column=2, sticky=tk.W)
    tk.Spinbox(magnitude_frame, from_=-5, to=15, textvariable=max_magnitude).grid(row=0, column=3, sticky=tk.W)


    (tk.Label(left_frame, text="===== Define the location ======", pady=5)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.W))

    # RA
    (tk.Label(left_frame, text="Center RA[h]", pady=5)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.W))
    (tk.Scale(left_frame, from_=0, to=24, resolution=1/64, orient=tk.HORIZONTAL, variable=RACentor)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.EW))
    
    # DEC
    (tk.Label(left_frame, text="Center location DEC[deg]", pady=5)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.W))
    (tk.Scale(left_frame, from_=-180, to=180, digits=1, orient=tk.HORIZONTAL, variable=DECCentor)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.EW))
    
    # RA
    (tk.Label(left_frame, text="Size RA[h]", pady=5)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.W))
    (tk.Scale(left_frame, from_=0, to=24, resolution=1/60, orient=tk.HORIZONTAL, variable=RASize)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.EW))
    
    # DEC
    (tk.Label(left_frame, text="Size DEC[deg]", pady=5)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.W))
    (tk.Scale(left_frame, from_=-180, to=180, digits=1, orient=tk.HORIZONTAL, variable=DECSize)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.EW))
    
    # Degree
    (tk.Label(left_frame, text="Rotation Angle[deg]", pady=5)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.W))
    (tk.Scale(left_frame, from_=-180, to=180, digits=1, orient=tk.HORIZONTAL, variable=deg)
     .grid(row=(settings_current_row := settings_current_row + 1), sticky=tk.EW))
    

    # -ImageDisplay
    image_label = tk.Label(center_frame, bg="dimgray")
    image_label.bind("<Configure>", on_image_label_size_changed)
    image_label.grid(sticky=tk.NSEW)

    # Buttons
    (tk.Button(right_frame, text="Open Picture", fg="blue", command=open_picture)
     .grid(row=0, column=0, sticky=tk.EW))
    (tk.Button(right_frame, text="Export CSV", fg="blue", command=export_csv)
     .grid(row=1, column=0, sticky=tk.EW))

    # Status Bar
    status_label = tk.Label(main_window, relief=tk.SUNKEN, anchor=tk.W)
    status_label.grid(row=1, column=0, columnspan=3, sticky=tk.EW)

    # Set default UI Values
    display_mode.set(0)
    star_num.set(300)
    draw_stars_blobs.set(1)
    threshold_min.set(10)
    blob_min_area.set(4)
    blob_max_area.set(100)
    min_dist_between_blobs.set(2)
    min_magnitude.set(1)
    max_magnitude.set(6)
    RACentor.set(3)
    DECCentor.set(-15)
    RASize.set(1)
    DECSize.set(15)
    deg.set(45)


build_ui()

# autoStart
input_image = cv.imread(input_fileName)
on_starnum_change(300)

show_frames()
main_window.mainloop()
