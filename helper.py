import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_to_grayscale(image):
    # Check if the image is already grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image
    elif len(image.shape) == 2:
        return image
    else:
        raise ValueError("Invalid image format")

def apply_gaussian_blur(gray_image, kernel_size=5):
    blurred_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
    return blurred_image

def apply_canny_edge_detection(gray_image, threshold1=100, threshold2=200):
    edges = cv2.Canny(gray_image, threshold1, threshold2)
    return edges

def apply_sobel_edge_detection(gray_image):
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    sobel_edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return sobel_edges

def apply_prewitt_edge_detection(gray_image):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    grad_x = cv2.filter2D(gray_image, -1, kernelx)
    grad_y = cv2.filter2D(gray_image, -1, kernely)
    prewitt_edges = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
    return prewitt_edges

def apply_roberts_edge_detection(gray_image):
    kernelx = np.array([[1, 0], [0, -1]])
    kernely = np.array([[0, 1], [-1, 0]])
    grad_x = cv2.filter2D(gray_image, -1, kernelx)
    grad_y = cv2.filter2D(gray_image, -1, kernely)
    roberts_edges = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
    return roberts_edges

def find_contours(edge_image):
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_bounding_boxes(image, contours):
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

def process_image(image, method='canny'):
    gray_image = convert_to_grayscale(image)
    blurred_image = apply_gaussian_blur(gray_image)

    if method == 'canny':
        edges = apply_canny_edge_detection(blurred_image, threshold1=50, threshold2=150)
    elif method == 'sobel':
        edges = apply_sobel_edge_detection(blurred_image)
    elif method == 'prewitt':
        edges = apply_prewitt_edge_detection(blurred_image)
    elif method == 'roberts':
        edges = apply_roberts_edge_detection(blurred_image)
    else:
        raise ValueError("Method should be 'canny', 'sobel', 'prewitt', or 'roberts'")

    contours = find_contours(edges)
    result_image = draw_bounding_boxes(image.copy(), contours)

    return cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), gray_image, edges

def display_images(original_image, gray_image, edge_image, result_image):
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale Image')

    plt.subplot(2, 2, 3)
    plt.imshow(edge_image, cmap='gray')
    plt.title('Edge Detected Image')

    plt.subplot(2, 2, 4)
    plt.imshow(result_image)
    plt.title('Bounding Boxes')

    plt.show()

# Example usage
# image_path = filedialog.askopenfilename()  # Replace with your image path
# original_image = cv2.imread(image_path)
# result_image, gray_image, edges = process_image(original_image, method='canny')
# display_images(original_image, gray_image, edges, result_image)
