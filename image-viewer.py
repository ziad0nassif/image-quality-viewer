import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QSlider,QComboBox, QDoubleSpinBox, QMessageBox)
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.roi_start = None
        self.roi_end = None
        self.roi1 = None
        self.roi2 = None
        self.roi3 = None
        self.current_roi = None
        self.setMinimumSize(400, 400)
        self.setText("No Image Loaded")
        self.is_active = False  # Add flag to track if this viewport is active
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.pixmap() is not None:
            self.is_active = True  # Set this viewport as active
            # Deactivate other viewports
            if isinstance(self.parent(), QWidget):
                for sibling in self.parent().findChildren(ImageLabel):
                    if sibling != self:
                        sibling.is_active = False
            self.roi_start = event.pos()
            self.roi_end = None
            self.update()
            
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.is_active:
            self.roi_end = event.pos()
            self.update()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.roi_start and self.roi_end and self.is_active:
            if not self.roi1:
                self.roi1 = QRect(self.roi_start, self.roi_end)
                self.current_roi = 1
            elif not self.roi2:
                self.roi2 = QRect(self.roi_start, self.roi_end)
                self.current_roi = 2
            elif not self.roi3:
                self.roi3 = QRect(self.roi_start, self.roi_end)
                self.current_roi = 3
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.pixmap() is not None:  # Only paint if there's an image
            painter = QPainter(self)
            
            # Draw current selection
            if self.roi_start and self.roi_end and self.is_active:
                painter.setPen(QPen(Qt.red, 2))
                painter.drawRect(QRect(self.roi_start, self.roi_end))
            
            # Draw existing ROIs
            if self.roi1:
                painter.setPen(QPen(Qt.blue, 2))
                painter.drawRect(self.roi1)
                painter.drawText(self.roi1.topLeft(), "ROI 1")
            
            if self.roi2:
                painter.setPen(QPen(Qt.green, 2))
                painter.drawRect(self.roi2)
                painter.drawText(self.roi2.topLeft(), "ROI 2")
            
            if self.roi3:
                painter.setPen(QPen(Qt.yellow, 2))
                painter.drawRect(self.roi3)
                painter.drawText(self.roi3.topLeft(), "ROI 3")

    def clear_rois(self):
        self.roi1 = None
        self.roi2 = None
        self.roi3 = None
        self.current_roi = None
        self.roi_start = None
        self.roi_end = None
        self.is_active = False
        self.update()

class ImageProcessor(QMainWindow):
   
    def __init__(self):
        super().__init__()
        self.initUI()
        self.original_image = None
        self.current_image = None
        self.output1_image = None
        self.output2_image = None
        self.is_grayscale = False
        self.active_viewport = None  # Track which viewport is active
        # Store slider values but don't apply automatically
        self.brightness_value = 0
        self.contrast_value = 0
        
        # Update stored values when sliders change
        self.brightness_slider.valueChanged.connect(self.update_brightness_value)
        self.contrast_slider.valueChanged.connect(self.update_contrast_value)

    def initUI(self):
        self.setWindowTitle('Image Processing Application')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        
        # Create image display area
        image_layout = QHBoxLayout()
        
        # Input viewport
        self.input_label = ImageLabel()
        self.input_label.setFixedSize(600, 600)
        self.input_label.setAlignment(Qt.AlignCenter)
        self.input_label.mouseDoubleClickEvent = lambda e: self.show_histogram(self.input_label)
        
        # Output1 viewport
        self.output1_label = ImageLabel()
        self.output1_label.setFixedSize(600, 600)
        self.output1_label.setAlignment(Qt.AlignCenter)
        self.output1_label.mouseDoubleClickEvent = lambda e: self.show_histogram(self.output1_label)
        
        # Output2 viewport
        self.output2_label = ImageLabel()
        self.output2_label.setFixedSize(600, 600)
        self.output2_label.setAlignment(Qt.AlignCenter)
        self.output2_label.mouseDoubleClickEvent = lambda e: self.show_histogram(self.output2_label)
        
        image_layout.addWidget(self.input_label)
        image_layout.addWidget(self.output1_label)
        image_layout.addWidget(self.output2_label)
        
        # Controls
        controls_layout = QHBoxLayout()

        # File operations
        self.load_btn = QPushButton('Load Image')
        self.load_btn.clicked.connect(self.load_image)

        # Image processing operations
        self.operation_combo = QComboBox()
        operations = ['Zoom', 'Add Noise', 'Denoise', 'Filters', 'Contrast Adjustment']
        self.operation_combo.addItems(operations)

        # Sub-operations
        self.sub_operation_combo = QComboBox()
        self.operation_combo.currentTextChanged.connect(self.update_sub_operations)

        # Parameters
        self.param_spinbox = QDoubleSpinBox()
        self.param_spinbox.setRange(0.1, 10.0)
        self.param_spinbox.setSingleStep(0.1)
        self.param_spinbox.setValue(1.0)

        # Status label to show if image is grayscale or color
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: blue;")

        # Apply buttons
        self.apply_btn = QPushButton('Apply to Output1')
        self.apply_btn.clicked.connect(lambda: self.apply_operation(target='output1'))

        self.apply_to_output1_btn = QPushButton('Apply to Output2')
        self.apply_to_output1_btn.clicked.connect(lambda: self.apply_operation(target='output2'))

        # Measurement buttons
        measure_layout = QHBoxLayout()
        self.snr_btn = QPushButton('Measure SNR')
        self.snr_btn.clicked.connect(self.measure_snr)
        self.cnr_btn = QPushButton('Measure CNR')
        self.cnr_btn.clicked.connect(self.measure_cnr)
        self.clear_roi_btn = QPushButton('Clear ROIs')
        self.clear_roi_btn.clicked.connect(self.clear_rois)

        measure_layout.addWidget(self.snr_btn)
        measure_layout.addWidget(self.cnr_btn)
        measure_layout.addWidget(self.clear_roi_btn)

        self.reset_btn = QPushButton('Reset Outputs')
        self.reset_btn.clicked.connect(self.reset_outputs)
        measure_layout.addWidget(self.reset_btn)

        # Add all controls to the main controls layout
        controls_layout.addWidget(self.load_btn)
        controls_layout.addWidget(self.operation_combo)
        controls_layout.addWidget(self.sub_operation_combo)
        controls_layout.addWidget(self.param_spinbox)
        controls_layout.addWidget(self.status_label)
        controls_layout.addWidget(self.apply_btn)
        controls_layout.addWidget(self.apply_to_output1_btn)
        controls_layout.addLayout(measure_layout)  # Add the measurement buttons layout
        
        layout.addLayout(image_layout)
        layout.addLayout(controls_layout)
        main_widget.setLayout(layout)
        
        self.update_sub_operations()


        bright_contrast_layout = QHBoxLayout()
    
        # Brightness slider
        brightness_label = QLabel("Brightness:")
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setTickPosition(QSlider.TicksBelow)
        
        # Contrast slider
        contrast_label = QLabel("Contrast:")
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(-100)
        self.contrast_slider.setMaximum(100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.setTickPosition(QSlider.TicksBelow)


        bright_contrast_layout.addWidget(brightness_label)
        bright_contrast_layout.addWidget(self.brightness_slider)
        bright_contrast_layout.addWidget(contrast_label)
        bright_contrast_layout.addWidget(self.contrast_slider)
        
        layout.addLayout(bright_contrast_layout)




        
    def update_sub_operations(self):
        self.sub_operation_combo.clear()
        operation = self.operation_combo.currentText()
        
        if operation == 'Zoom':
            self.sub_operation_combo.addItems(['Nearest-Neighbor', 'Linear', 'Bilinear', 'Cubic'])
        elif operation == 'Add Noise':
            self.sub_operation_combo.addItems(['Gaussian', 'Salt & Pepper', 'Speckle'])
        elif operation == 'Denoise':
            self.sub_operation_combo.addItems(['Median', 'Gaussian', 'Non-local Means'])
        elif operation == 'Filters':
            self.sub_operation_combo.addItems(['Lowpass', 'Highpass'])
        elif operation == 'Contrast Adjustment':
            self.sub_operation_combo.addItems(['Histogram Equalization', 'CLAHE', 'Adaptive Contrast'])

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', 
                                            '', "Image files (*.jpg *.png *.bmp)")
        if fname:
            # Read image and determine if it's grayscale
            temp_image = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            if temp_image is None:
                QMessageBox.critical(self, "Error", "Failed to load image.")
                return
                
            # Handle different image types
            if len(temp_image.shape) == 2:
                # Single channel grayscale
                self.original_image = temp_image
                self.is_grayscale = True
                self.status_label.setText("Grayscale Image")
            elif len(temp_image.shape) == 3:
                if temp_image.shape[2] == 1:
                    # Single channel with extra dimension
                    self.original_image = temp_image[:,:,0]
                    self.is_grayscale = True
                    self.status_label.setText("Grayscale Image")
                elif temp_image.shape[2] == 3:
                    # Check if RGB channels are equal (grayscale)
                    b = temp_image[:,:,0]
                    g = temp_image[:,:,1]
                    r = temp_image[:,:,2]
                    if np.array_equal(b, g) and np.array_equal(g, r):
                        self.original_image = b
                        self.is_grayscale = True
                        self.status_label.setText("Grayscale Image")
                    else:
                        self.original_image = temp_image
                        self.is_grayscale = False
                        self.status_label.setText("Color Image")
                elif temp_image.shape[2] == 4:
                    # Handle RGBA images by converting to RGB
                    self.original_image = cv2.cvtColor(temp_image, cv2.COLOR_BGRA2BGR)
                    self.is_grayscale = False
                    self.status_label.setText("Color Image")
                
            self.current_image = self.original_image.copy()
            self.display_image(self.input_label, self.current_image)
            
            # Reset output images
            self.output1_image = None
            self.output2_image = None
            self.output1_label.setText("No Image")
            self.output2_label.setText("No Image")

    def reset_outputs(self):
        if self.original_image is not None:
            # Reset all images to original
            self.current_image = self.original_image.copy()
            self.output1_image = self.original_image.copy()
            self.output2_image = self.original_image.copy()
            
            # Reset brightness/contrast sliders
            self.brightness_slider.setValue(0)
            self.contrast_slider.setValue(0)
            
            # Update displays
            self.display_image(self.input_label, self.current_image)
            self.display_image(self.output1_label, self.output1_image)
            self.display_image(self.output2_label, self.output2_image)
            
            # Clear ROIs
            self.clear_rois()
            



    def toggle_grayscale(self, state):
        if self.current_image is not None:
            if state == Qt.Checked and not self.is_grayscale:
                if len(self.current_image.shape) == 3:
                    self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
                    self.is_grayscale = True
            elif state == Qt.Unchecked and self.is_grayscale:
                if len(self.original_image.shape) == 3:
                    self.current_image = self.original_image.copy()
                    self.is_grayscale = False
            
            self.display_image(self.input_label, self.current_image)



    def display_image(self, label, image):
        if image is not None:
            if len(image.shape) == 2:  # Grayscale image
                height, width = image.shape
                bytes_per_line = width
                qt_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:  # Color image (BGR)
                height, width, channels = image.shape
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                bytes_per_line = channels * width
                qt_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                
            label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def show_histogram(self, label):
        if label == self.input_label and self.current_image is not None:
            image = self.current_image
        elif label == self.output1_label and self.output1_image is not None:
            image = self.output1_image
        elif label == self.output2_label and self.output2_image is not None:
            image = self.output2_image
        else:
            return

        plt.figure()
        if len(image.shape) == 2:  # Grayscale image
            plt.hist(image.ravel(), 256, [0, 256], color='black')
        else:  # Color image (RGB)
            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                plt.hist(image[:, :, i].ravel(), 256, [0, 256], color=color, alpha=0.5)
        plt.title('Image Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()

    def apply_brightness_contrast(self, target):
            if target == 'output1':
                source_image = self.current_image
            elif target == 'output2':
                source_image = self.output1_image
            else:
                return

            if source_image is None:
                return

            # Create a copy of the source image
            adjusted = source_image.copy().astype(float)

            # Apply contrast
            contrast = (self.contrast_slider.value() + 100) / 100.0  # Convert to factor
            adjusted = adjusted * contrast

            # Apply brightness
            brightness = self.brightness_slider.value()
            adjusted = adjusted + brightness

            # Clip values to valid range
            adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

            # Update the appropriate output
            if target == 'output1':
                self.output1_image = adjusted
                self.display_image(self.output1_label, adjusted)
            else:
                self.output2_image = adjusted
                self.display_image(self.output2_label, adjusted)

                    
    def update_brightness_value(self, value):
            self.brightness_value = value

    def update_contrast_value(self, value):
        self.contrast_value = value

    def apply_operation(self, target):
        if target == 'output1':
            source_image = self.current_image
        else:
            source_image = self.output1_image if self.output1_image is not None else self.current_image

        if source_image is None:
            QMessageBox.warning(self, "Error", "No source image available.")
            return

        operation = self.operation_combo.currentText()
        sub_operation = self.sub_operation_combo.currentText()
        param = self.param_spinbox.value()

        try:
            # Apply selected operation
            if operation == 'Zoom':
                result = self.apply_zoom(source_image, sub_operation, param)
            elif operation == 'Add Noise':
                result = self.apply_noise(source_image, sub_operation)
            elif operation == 'Denoise':
                result = self.apply_denoise(source_image, sub_operation)
            elif operation == 'Filters':
                result = self.apply_filter(source_image, sub_operation)
            elif operation == 'Contrast Adjustment':
                result = self.apply_contrast_adjustment(source_image, sub_operation)

            # Apply brightness and contrast
            adjusted = result.copy().astype(float)
            contrast = (self.contrast_value + 100) / 100.0
            adjusted = adjusted * contrast
            adjusted = adjusted + self.brightness_value
            result = np.clip(adjusted, 0, 255).astype(np.uint8)

            # Update appropriate output and display
            if target == 'output1':
                self.output1_image = result.copy()
                self.display_image(self.output1_label, self.output1_image)
            else:
                self.output2_image = result.copy()
                self.display_image(self.output2_label, self.output2_image)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Operation failed: {str(e)}")


            
    def apply_zoom(self, image, method, factor):
        if image is None:
            return None
        
        h, w = image.shape

        if factor >= 1:  # Zoom In
            # Calculate the size of the cropped region
            crop_h = int(h / factor)
            crop_w = int(w / factor)

            # Calculate the center point
            center_y = h // 2
            center_x = w // 2

            # Calculate the crop boundaries
            start_y = max(0, center_y - (crop_h // 2))
            start_x = max(0, center_x - (crop_w // 2))
            end_y = min(h, start_y + crop_h)
            end_x = min(w, start_x + crop_w)

            # Crop the region around the center
            cropped = image[start_y:end_y, start_x:end_x]

            # Interpolate the cropped image back to the original size
            if method == 'Nearest-Neighbor':
                return self.nearest_neighbor_zoom(cropped, w, h)
            elif method == 'Linear':
                return self.linear_zoom(cropped, w, h)
            elif method == 'Bilinear':
                return self.bilinear_zoom(cropped, w, h)
            else:  # Cubic
                return self.cubic_zoom(cropped, w, h)
        else:  # Zoom Out
            # Calculate the new dimensions
            new_h = int(h * factor)
            new_w = int(w * factor)

            # Resize the image using the selected interpolation method
            if method == 'Nearest-Neighbor':
                return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            elif method == 'Linear':
                return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            elif method == 'Bilinear':  # OpenCV treats 'Bilinear' as 'Linear'
                return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:  # Cubic
                return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


    def nearest_neighbor_zoom(self, image, target_width, target_height):
        """
        Nearest neighbor interpolation - maintains sharp pixels
        """
        h, w = image.shape
        output = np.zeros((target_height, target_width), dtype=np.uint8)
        
        # Calculate scaling ratios
        x_ratio = w / target_width
        y_ratio = h / target_height
        
        for i in range(target_height):
            for j in range(target_width):
                # Find nearest source pixel
                src_x = min(w - 1, int(j * x_ratio))
                src_y = min(h - 1, int(i * y_ratio))
                output[i, j] = image[src_y, src_x]
        
        return output

    def linear_zoom(self, image, target_width, target_height):
        """
        Linear interpolation - simple smooth interpolation
        """
        h, w = image.shape
        output = np.zeros((target_height, target_width), dtype=np.uint8)
        
        x_ratio = float(w - 1) / target_width
        y_ratio = float(h - 1) / target_height
        
        for i in range(target_height):
            for j in range(target_width):
                # Get the four surrounding pixels
                x = int(x_ratio * j)
                y = int(y_ratio * i)
                
                # Get linear interpolation weights
                x_diff = (x_ratio * j) - x
                y_diff = (y_ratio * i) - y
                
                # Linear interpolation
                if x + 1 < w and y + 1 < h:
                    pixel = (1 - x_diff) * (1 - y_diff) * image[y, x] + \
                        x_diff * (1 - y_diff) * image[y, x + 1]
                    output[i, j] = np.clip(pixel, 0, 255)
                else:
                    output[i, j] = image[min(y, h-1), min(x, w-1)]
        
        return output

    def bilinear_zoom(self, image, target_width, target_height):
        """
        Bilinear interpolation - smoother interpolation using 4 pixels
        """
        h, w = image.shape
        output = np.zeros((target_height, target_width), dtype=np.uint8)
        
        x_ratio = float(w - 1) / target_width
        y_ratio = float(h - 1) / target_height
        
        for i in range(target_height):
            for j in range(target_width):
                x = int(x_ratio * j)
                y = int(y_ratio * i)
                
                x_diff = (x_ratio * j) - x
                y_diff = (y_ratio * i) - y
                
                if x + 1 < w and y + 1 < h:
                    # Get the four surrounding pixels
                    a = image[y, x]
                    b = image[y, x + 1]
                    c = image[y + 1, x]
                    d = image[y + 1, x + 1]
                    
                    # Bilinear interpolation
                    pixel = a * (1 - x_diff) * (1 - y_diff) + \
                        b * x_diff * (1 - y_diff) + \
                        c * y_diff * (1 - x_diff) + \
                        d * x_diff * y_diff
                        
                    output[i, j] = np.clip(pixel, 0, 255)
                else:
                    output[i, j] = image[min(y, h-1), min(x, w-1)]
        
        return output

    def cubic_zoom(self, image, target_width, target_height):
        """
        Cubic interpolation - highest quality interpolation using 16 pixels
        """
        def cubic_weight(x):
            abs_x = abs(x)
            if abs_x <= 1:
                return 1 - 2 * abs_x**2 + abs_x**3
            elif abs_x < 2:
                return 4 - 8 * abs_x + 5 * abs_x**2 - abs_x**3
            return 0
        
        h, w = image.shape
        output = np.zeros((target_height, target_width), dtype=np.uint8)
        
        x_ratio = float(w - 1) / target_width
        y_ratio = float(h - 1) / target_height
        
        for i in range(target_height):
            for j in range(target_width):
                x = x_ratio * j
                y = y_ratio * i
                
                x_int = int(x)
                y_int = int(y)
                
                # Calculate the cubic interpolation
                pixel = 0
                weightsum = 0
                
                for ii in range(max(0, y_int-1), min(h, y_int+3)):
                    for jj in range(max(0, x_int-1), min(w, x_int+3)):
                        dx = jj - x
                        dy = ii - y
                        weight = cubic_weight(dx) * cubic_weight(dy)
                        pixel += image[ii, jj] * weight
                        weightsum += weight
                
                if weightsum != 0:
                    pixel = pixel / weightsum
                
                output[i, j] = np.clip(pixel, 0, 255)
        
        return output

    def apply_noise(self, image, noise_type):
        if len(image.shape) == 2:  # Grayscale image
            if noise_type == 'Gaussian':
                intensity = np.std(image) * 0.8  # Noise intensity based on image std
                noise = np.random.normal(0, intensity, image.shape)
                noisy = image + noise

                return np.clip(noisy, 0, 255).astype(np.uint8)
            
            elif noise_type == 'Salt & Pepper':
                noisy = image.copy()
                prob = 0.05  # Fixed probability
                black = np.random.random(image.shape) < prob
                white = np.random.random(image.shape) < prob
                noisy[black] = 0
                noisy[white] = 255
                return noisy
            
            else:  # Speckle
                intensity = np.std(image) * 0.01  # Noise intensity based on image std
                noise = np.random.normal(0, intensity, image.shape)
                noisy = image + image * noise
                return np.clip(noisy, 0, 255).astype(np.uint8)
            
        else:  # Color image (RGB)
            noisy = np.zeros_like(image)
            for i in range(3):  # Apply noise to each channel
                noisy[:, :, i] = self.apply_noise(image[:, :, i], noise_type)
            return noisy
    
            
    def apply_denoise(self, image, method):

        if len(image.shape) == 2:  # Grayscale image
            if method == 'Median':
                ksize = max(3, min(image.shape[:2]) // 20)  # Kernel size based on image size
                ksize = ksize if ksize % 2 == 1 else ksize + 1  # Ensure odd
                return cv2.medianBlur(image, ksize)
            
            elif method == 'Gaussian':
                ksize = max(3, min(image.shape[:2]) // 20)  # Kernel size based on image size
                ksize = ksize if ksize % 2 == 1 else ksize + 1  # Ensure odd
                return cv2.GaussianBlur(image, (ksize, ksize), 0)
            
            else:  # Non-local Means
                strength = np.std(image) * 0.1  # Denoising strength based on image std
                return cv2.fastNlMeansDenoising(image, None, strength)
                
        else:  # Color image (RGB)
            denoised = np.zeros_like(image)
            for i in range(3):  # Apply denoising to each channel
                if method == 'Median':
                    ksize = max(3, min(image.shape[:2]) // 20)  # Kernel size based on image size
                    ksize = ksize if ksize % 2 == 1 else ksize + 1  # Ensure odd
                    denoised[:, :, i] = cv2.medianBlur(image[:, :, i], ksize)
                elif method == 'Gaussian':
                    ksize = max(3, min(image.shape[:2]) // 20)  # Kernel size based on image size
                    ksize = ksize if ksize % 2 == 1 else ksize + 1  # Ensure odd
                    denoised[:, :, i] = cv2.GaussianBlur(image[:, :, i], (ksize, ksize), 0)
                else:  # Non-local Means
                    strength = np.std(image[:, :, i]) * 0.1  # Denoising strength based on channel std
                    denoised[:, :, i] = cv2.fastNlMeansDenoising(image[:, :, i], None, strength)
            return denoised
    
            
    def apply_filter(self, image, filter_type):
        # Hardcoded kernel size
        ksize = max(3, min(image.shape[:2]) // 20)  # Kernel size based on image size
        ksize = ksize if ksize % 2 == 1 else ksize + 1  # Ensure odd

        if len(image.shape) == 2:  # Grayscale image
            if filter_type == 'Lowpass':
                return cv2.GaussianBlur(image, (ksize, ksize), 0)
            else:  # Highpass
                lowpass = cv2.GaussianBlur(image, (ksize, ksize), 0)
                return cv2.subtract(image, lowpass)
        else:  # Color image (RGB)
            filtered = np.zeros_like(image)
            for i in range(3):  # Apply filtering to each channel
                if filter_type == 'Lowpass':
                    filtered[:, :, i] = cv2.GaussianBlur(image[:, :, i], (ksize, ksize), 0)
                else:  # Highpass
                    lowpass = cv2.GaussianBlur(image[:, :, i], (ksize, ksize), 0)
                    filtered[:, :, i] = cv2.subtract(image[:, :, i], lowpass)
            return filtered
            
    def apply_contrast_adjustment(self, image, method):
        if len(image.shape) == 2:  # Grayscale
            if method == 'Histogram Equalization':
                return cv2.equalizeHist(image)
            elif method == 'CLAHE':
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                return clahe.apply(image)
            else:  # Adaptive Contrast Enhancement
                # Calculate local mean and std
                local_mean = cv2.GaussianBlur(image, (25,25), 0)
                local_std = np.sqrt(cv2.GaussianBlur(np.square(image - local_mean), (25,25), 0))
                
                # Enhance contrast based on local statistics
                enhanced = ((image - local_mean) / (local_std + 1)) * 50 + 128
                return np.clip(enhanced, 0, 255).astype(np.uint8)
        else:  # Color
            adjusted = np.zeros_like(image)
            for i in range(3):
                channel = image[:,:,i]
                if method == 'Histogram Equalization':
                    adjusted[:,:,i] = cv2.equalizeHist(channel)
                elif method == 'CLAHE':
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    adjusted[:,:,i] = clahe.apply(channel)
                else:
                    local_mean = cv2.GaussianBlur(channel, (25,25), 0)
                    local_std = np.sqrt(cv2.GaussianBlur(np.square(channel - local_mean), (25,25), 0))
                    enhanced = ((channel - local_mean) / (local_std + 1)) * 50 + 128
                    adjusted[:,:,i] = np.clip(enhanced, 0, 255).astype(np.uint8)
            return adjusted


    def measure_snr(self):
        # Get the active viewport and corresponding image
        viewport, image = self.get_active_viewport_and_image()
        if viewport is None or image is None:
            QMessageBox.warning(self, "Warning", "Please select a viewport with an image")
            return
            
        if viewport.roi1 is None or viewport.roi2 is None:
            QMessageBox.warning(self, "Warning", 
                              "Please select two ROIs:\nROI 1: Signal region\nROI 2: Noise region")
            return
            
        # Get ROI coordinates
        roi1_rect = viewport.roi1
        roi2_rect = viewport.roi2
        
        # Extract ROI regions from the image
        if len(image.shape) == 2:  # Grayscale
            roi1_data = image[roi1_rect.top():roi1_rect.bottom(), 
                            roi1_rect.left():roi1_rect.right()]
            roi2_data = image[roi2_rect.top():roi2_rect.bottom(), 
                            roi2_rect.left():roi2_rect.right()]
        else:  # Color
            roi1_data = cv2.cvtColor(image[roi1_rect.top():roi1_rect.bottom(), 
                                          roi1_rect.left():roi1_rect.right()], 
                                   cv2.COLOR_BGR2GRAY)
            roi2_data = cv2.cvtColor(image[roi2_rect.top():roi2_rect.bottom(), 
                                          roi2_rect.left():roi2_rect.right()], 
                                   cv2.COLOR_BGR2GRAY)
        
        # Calculate SNR
        signal_mean = np.mean(roi1_data)
        noise_std = np.std(roi2_data)
        
        snr = signal_mean / noise_std if noise_std != 0 else float('inf')
        
        # Show results
        msg = (f'Signal Mean (ROI 1): {signal_mean:.2f}\n'
               f'Noise StdDev (ROI 2): {noise_std:.2f}\n'
               f'SNR: {snr:.2f}')
        QMessageBox.information(self, 'SNR Measurement', msg)

    def measure_cnr(self):
        # Get the active viewport and corresponding image
        viewport, image = self.get_active_viewport_and_image()
        if viewport is None or image is None:
            QMessageBox.warning(self, "Warning", "Please select a viewport with an image")
            return
            
        if viewport.roi1 is None or viewport.roi2 is None or viewport.roi3 is None:
            QMessageBox.warning(self, "Warning", 
                              "Please select three ROIs:\n"
                              "ROI 1: First region\n"
                              "ROI 2: Second region\n"
                              "ROI 3: Background noise region")
            return
            
        # Get ROI coordinates
        roi1_rect = viewport.roi1
        roi2_rect = viewport.roi2
        roi3_rect = viewport.roi3
        
        # Extract ROI regions from the image
        if len(image.shape) == 2:  # Grayscale
            roi1_data = image[roi1_rect.top():roi1_rect.bottom(), 
                            roi1_rect.left():roi1_rect.right()]
            roi2_data = image[roi2_rect.top():roi2_rect.bottom(), 
                            roi2_rect.left():roi2_rect.right()]
            roi3_data = image[roi3_rect.top():roi3_rect.bottom(), 
                            roi3_rect.left():roi3_rect.right()]
        else:  # Color
            roi1_data = cv2.cvtColor(image[roi1_rect.top():roi1_rect.bottom(), 
                                          roi1_rect.left():roi1_rect.right()], 
                                   cv2.COLOR_BGR2GRAY)
            roi2_data = cv2.cvtColor(image[roi2_rect.top():roi2_rect.bottom(), 
                                          roi2_rect.left():roi2_rect.right()], 
                                   cv2.COLOR_BGR2GRAY)
            roi3_data = cv2.cvtColor(image[roi3_rect.top():roi3_rect.bottom(), 
                                          roi3_rect.left():roi3_rect.right()], 
                                   cv2.COLOR_BGR2GRAY)
        
        # Calculate CNR
        mean1 = np.mean(roi1_data)
        mean2 = np.mean(roi2_data)
        noise_std = np.std(roi3_data)

        
        cnr = abs(mean1 - mean2) / noise_std if noise_std != 0 else float('inf')
        
        # Show results
        msg = (f'Region 1 Mean (ROI 1): {mean1:.2f}\n'
               f'Region 2 Mean (ROI 2): {mean2:.2f}\n'
               f'Background StdDev (ROI 3): {noise_std:.2f}\n'
               f'CNR: {cnr:.2f}')
        QMessageBox.information(self, 'CNR Measurement', msg)

    def get_active_viewport_and_image(self):
        """Returns the active viewport and its corresponding image."""
        # First check if any viewport has ongoing ROI selection
        for viewport, image in [
            (self.input_label, self.current_image),
            (self.output1_label, self.output1_image),
            (self.output2_label, self.output2_image)
        ]:
            if viewport.roi_start is not None or viewport.roi_end is not None:
                return viewport, image
        
        # Then check if any viewport has existing ROIs
        for viewport, image in [
            (self.input_label, self.current_image),
            (self.output1_label, self.output1_image),
            (self.output2_label, self.output2_image)
        ]:
            if viewport.roi1 is not None or viewport.roi2 is not None or viewport.roi3 is not None:
                return viewport, image
        
        # If no active viewport found, return None
        return None, None

    def clear_rois(self):
        """Clear ROIs from all viewports."""
        self.input_label.clear_rois()
        self.output1_label.clear_rois()
        self.output2_label.clear_rois()



if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = ImageProcessor()
    ex.show()
    sys.exit(app.exec_())
