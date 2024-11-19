import cv2
import numpy as np
import os
from tkinter import Tk, Label, Button, filedialog, messagebox

# Function to capture an image from the webcam
def capture_image():
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        messagebox.showerror("Error", "Camera not detected!")
        return

    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Failed to capture image!")
        cap.release()
        return

    # Show the captured image in a window
    cv2.imshow("Captured Image", frame)
    save_path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")],
        title="Save Captured Image"
    )
    if save_path:
        cv2.imwrite(save_path, frame)  # Save the image
        global captured_image_path
        captured_image_path = save_path
        messagebox.showinfo("Success", f"Image saved as {save_path}")

    cap.release()
    cv2.destroyAllWindows()

# Function to compare the captured image with all images in the "stored_data" folder
def compare_with_stored_data():
    try:
        # Folder containing stored images
        folder_path = "stored_data"
        if not os.path.exists(folder_path):
            messagebox.showerror("Error", f"Folder '{folder_path}' not found.")
            return

        # Ensure captured image exists
        if captured_image_path is None:
            messagebox.showerror("Error", "No captured image found. Please capture an image first.")
            return

        # Load the captured image
        captured_image = cv2.imread(captured_image_path, cv2.IMREAD_GRAYSCALE)
        captured_image = cv2.resize(captured_image, (200, 200))

        # Iterate through each image in the folder and compare
        best_match = None
        highest_similarity = 0

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                stored_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if stored_image is None:
                    continue

                # Resize stored image to match captured image size
                stored_image = cv2.resize(stored_image, (200, 200))

                # Calculate similarity using Mean Squared Error (MSE)
                mse = np.mean((captured_image - stored_image) ** 2)
                similarity = 100 - (mse / 255 * 100)

                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = file_name

        # Display the result
        if highest_similarity > 80:  # Threshold for originality
            messagebox.showinfo("Result", f"Match found: {best_match}\nSimilarity: {highest_similarity:.2f}%")
        else:
            messagebox.showwarning("Result", "No sufficient match found in the folder.")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Initialize the GUI
app = Tk()
app.title("Image Verification System")

Label(app, text="Image Verification System", font=("Arial", 16)).pack(pady=10)
Button(app, text="Capture Image", command=capture_image, font=("Arial", 14)).pack(pady=5)
Button(app, text="Compare with Stored Images Folder", command=compare_with_stored_data, font=("Arial", 14)).pack(pady=5)

# Global variable to store the path of the captured image
captured_image_path = None

app.mainloop()

