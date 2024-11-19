import cv2
import os
from tkinter import Tk, Label, Button, filedialog, messagebox, Canvas
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import cosine_similarity

# Paths to directories
CAPTURE_FOLDER = "Capture_data"
STORED_FOLDER = "stored_data"

# Ensure the capture folder exists
if not os.path.exists(CAPTURE_FOLDER):
    os.makedirs(CAPTURE_FOLDER)

# Function to capture an image from the webcam
def capture_image():
    global captured_image_path

    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        messagebox.showerror("Error", "Camera not detected!")
        return

    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Failed to capture image!")
        cap.release()
        return

    # Save the image to Capture_data folder
    captured_image_path = os.path.join(CAPTURE_FOLDER, "captured_image.jpg")
    cv2.imwrite(captured_image_path, frame)

    # Convert to a Tkinter-compatible format
    img = Image.open(captured_image_path)
    img_tk = ImageTk.PhotoImage(img)

    # Display the image on the Canvas
    canvas.delete("all")  # Clear the canvas
    canvas.config(width=img.width, height=img.height)
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image = img_tk

    messagebox.showinfo("Success", "Image captured successfully!")
    cap.release()

# Function to verify the captured image with stored images
def verify_image():
    if not captured_image_path:
        messagebox.showerror("Error", "No captured image to verify!")
        return

    # Load the captured image and convert it to grayscale
    captured_img = cv2.imread(captured_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Initialize variables for best match
    best_match_score = float('inf')
    best_match_filename = None

    # Loop through all images in the stored_data folder
    for filename in os.listdir(STORED_FOLDER):
        stored_image_path = os.path.join(STORED_FOLDER, filename)
        
        if os.path.isfile(stored_image_path):
            # Read and convert each stored image to grayscale
            stored_img = cv2.imread(stored_image_path, cv2.IMREAD_GRAYSCALE)
            
            # Compute the similarity score between the captured image and the stored image
            similarity_score = compare_images(captured_img, stored_img)
            
            if similarity_score < best_match_score:
                best_match_score = similarity_score
                best_match_filename = filename

    # If a match is found, display the result
    if best_match_filename:
        messagebox.showinfo("Verification Result", f"Image matches with: {best_match_filename}\nSimilarity Score: {best_match_score:.2f}")
    else:
        messagebox.showinfo("Verification Result", "No match found")
    # If a match is found, display the result
    if best_match_filename:
        messagebox.showinfo("Verification Result", f"Image matches with: {best_match_filename}\nSimilarity Score: {best_match_score:.2f}")
    else:
        messagebox.showinfo("Verification Result", "No match found")

# Function to compare two images using Structural Similarity Index (SSI)
from sklearn.metrics.pairwise import cosine_similarity

def compare_images(image1, image2):
    # Resize the images to the same size for comparison
    image1_resized = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

    # Flatten the images and calculate the cosine similarity between the two
    similarity_score = cosine_similarity(image1_resized.flatten().reshape(1, -1), image2.flatten().reshape(1, -1))

    # Return the similarity score (which is a 2D array, so access the value)
    return similarity_score[0][0]


# Function to save the captured image
def save_image():
    if not captured_image_path:
        messagebox.showerror("Error", "No captured image to save!")
        return

    save_path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")],
        title="Save Captured Image"
    )
    if save_path:
        cv2.imwrite(save_path, cv2.imread(captured_image_path))
        messagebox.showinfo("Success", f"Image saved as {save_path}")

# Function to show the captured image using Matplotlib
def show_image_matplotlib():
    if not captured_image_path:
        messagebox.showerror("Error", "No captured image to display!")
        return

    # Load and display the image using Matplotlib
    img = cv2.imread(captured_image_path)  # Read the captured image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format for Matplotlib

    plt.imshow(img_rgb)
    plt.axis("off")  # Turn off axes
    plt.title("Captured Image")
    plt.show()

# Initialize the Tkinter app
app = Tk()
app.title("Image Capture and Verification System")

# Tkinter GUI Elements
Label(app, text="Image Capture and Verification System", font=("Arial", 16)).pack(pady=10)

Button(app, text="Capture Image", command=capture_image, font=("Arial", 14)).pack(pady=5)
Button(app, text="Verify Image", command=verify_image, font=("Arial", 14)).pack(pady=5)
Button(app, text="Save Image", command=save_image, font=("Arial", 14)).pack(pady=5)
Button(app, text="Show Image in Matplotlib", command=show_image_matplotlib, font=("Arial", 14)).pack(pady=5)

# Canvas to display the captured image
canvas = Canvas(app, width=400, height=300, bg="gray")
canvas.pack(pady=10)

# Global variable to store the path of the captured image
captured_image_path = None

# Start the Tkinter main loop
app.mainloop()

