# Image Viewer Application

## Introduction
### This project is a simple image viewer application built with Python and PyQt5, designed to load and display images using a user-friendly graphical interface. It supports opening images, zooming, and provides a file dialog for selecting image files from the system. The Image Viewer Application allows users to open, view, and make some adjusments we will mention it later. It provides a straightforward interface to navigate through directories and select images for viewing. This project is a great example of using PyQt5 to create desktop GUI applications in Python.

Image of Applictaion:
<div>
  <img src =  "https://github.com/user-attachments/assets/6090a9c1-7f54-40bc-95b0-74d675778845">
</div>

## Features

• Open and view 2D image in one input viewport.

• Support gray scale or colorful images to try adjusments on.

• Make changes and view the result on one of two available viewports.

• The user can apply one change on the input, show the result in output1 then apply
another change on output1 and show the results in output2.

• For each image (input1 or output2), the user can show its histogram at any point (by double
clicking)

• The user can apply the below functionalities:

• Resolution: Zoom in/out with different factors. When zooming in, the result image can
be interpolated using nearest-neighbor, linear, bilinear, cubic interpolation.

• Allow the user to measure the SNR or CNR by putting two ot three ROIs and measure.

• Apply 3 different types of noises on the image like 'Gaussian' or 'Salt & Pepper' or 'Speckle'.

• Apply 3 different types of denoising techniques/filters on the image like 'Median' or 'Gaussian' or 'Non-local Means'.

• Apply Lowpass and highpass filters.

• Change the brightness and cotrast of the image.

• Apply 3 different contrast adjustment for improving the CNR likehistogram
equalization or CLAHE or Adaptive Contrast.

## Requirements
To run this application, make sure the following are installed:

[requirements.txt](https://github.com/user-attachments/files/18285030/requirements.txt) 


## Usage
Upon running the application, you can use the "Load Image" button to choose an image from your system.

The image will be displayed at the selected scale, and you can easily navigate through the Buttons.


## Logging
The program logs user interactions and critical steps, aiding in debugging and problem resolution. Log files are generated to provide insights into the development process.

### Feel free to fork this repository, make improvements, and submit a pull request if you have any enhancements or bug fixes.


