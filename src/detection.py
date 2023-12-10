# Importing all the necessary libraries

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from skimage import exposure, filters
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose, Dense, Reshape, Flatten


# Step 1: Load the Dataset
data_dir = "../data/HYPODENSITY-DATA"
case_ids = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

# Load all 8 cases
ncct_images = []
roi_masks = []

for case_id in case_ids:
    ncct_image_path = os.path.join(data_dir, case_id, f"{case_id}_NCCT.nii.gz")
    roi_mask_path = os.path.join(data_dir, case_id, f"{case_id}_ROI.nii.gz")

    ncct_image = nib.load(ncct_image_path).get_fdata()
    roi_mask = nib.load(roi_mask_path).get_fdata()

    # Preprocess NCCT image (if needed)
    ncct_image = (ncct_image - np.min(ncct_image)) / (np.max(ncct_image) - np.min(ncct_image))

    ncct_images.append(np.expand_dims(ncct_image, axis=-1))
    roi_masks.append(np.expand_dims(roi_mask, axis=-1))

# -------------------------------------------------------------------------------------------------------------

# # Iterate through all cases
# for case_index in range(len(ncct_images)):
#     ncct_image = ncct_images[case_index]
#     roi_mask = roi_masks[case_index]
#
#     # Check the depth size for the current case
#     depth_size = ncct_image.shape[2]  # Assuming depth is along the third axis
#
#     # Visualize each slice along with the ROI mask
#     for slice_index in range(depth_size):
#         plt.figure(figsize=(12, 6))
#
#         # Visualize the NCCT image
#         plt.subplot(1, 2, 1)
#         plt.imshow(ncct_image[:, :, slice_index, 0], cmap='gray')
#         plt.title(f"NCCT Image - Case {case_index + 1}, Slice {slice_index}")
#
#         # Visualize the ROI mask
#         plt.subplot(1, 2, 2)
#         plt.imshow(roi_mask[:, :, slice_index, 0], cmap='viridis')
#         plt.title(f"Hypodensity Mask - Case {case_index + 1}, Slice {slice_index}")
#
#         plt.show()

# ---------------------------------------------------------------------------------------------------------------------

def preprocess_image(image):
    # Normalize pixel values to the range [0, 1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Apply Gaussian smoothing to reduce noise
    smoothed_image = filters.gaussian(image, sigma=1)

    # Enhance contrast using histogram equalization
    enhanced_image = exposure.equalize_hist(smoothed_image)

    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = exposure.equalize_adapthist(enhanced_image, clip_limit=0.03)

    # You can add more preprocessing steps here based on your requirements

    return clahe

# Example usage:
preprocessed_ncct_images = [preprocess_image(image) for image in ncct_images]

# Find the maximum number of slices
max_slices = max(image.shape[2] for image in preprocessed_ncct_images)

# Adjust all arrays to have the same number of slices
for i in range(len(preprocessed_ncct_images)):
    ncct_image = preprocessed_ncct_images[i]
    roi_mask = roi_masks[i]

    # Check if adjustment is needed
    if ncct_image.shape[2] < max_slices:
        # Pad slices
        pad_slices = max_slices - ncct_image.shape[2]
        pad_start = pad_slices // 2
        pad_end = pad_slices - pad_start
        ncct_image = np.concatenate([np.zeros_like(ncct_image[:, :, :pad_start]),
                                     ncct_image,
                                     np.zeros_like(ncct_image[:, :, :pad_end])], axis=2)

        roi_mask = np.concatenate([np.zeros_like(roi_mask[:, :, :pad_start]),
                                   roi_mask,
                                   np.zeros_like(roi_mask[:, :, :pad_end])], axis=2)

    elif ncct_image.shape[2] > max_slices:
        # Trim slices
        slice_diff = ncct_image.shape[2] - max_slices
        start_slice = slice_diff // 2
        end_slice = ncct_image.shape[2] - (slice_diff - start_slice)
        ncct_image = ncct_image[:, :, start_slice:end_slice]
        roi_mask = roi_mask[:, :, start_slice:end_slice]

    # Update the preprocessed_ncct_images and roi_masks lists
    preprocessed_ncct_images[i] = ncct_image
    roi_masks[i] = roi_mask

# # Verify that all arrays have the same number of slices
# for i, (ncct_image, roi_mask) in enumerate(zip(preprocessed_ncct_images, roi_masks)):
#     print(f"Case {i + 1} - NCCT Image Shape: {ncct_image.shape}, ROI Mask Shape: {roi_mask.shape}")

#----------------------------------------------------------------------------------------------------------------------

# Combine preprocessed data into arrays
X = np.array(preprocessed_ncct_images)
y = np.array(roi_masks)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# # Verify the shapes of the sets
# print("Training data shapes:", X_train.shape, y_train.shape)
# print("Validation data shapes:", X_val.shape, y_val.shape)
# print("Test data shapes:", X_test.shape, y_test.shape)

#-------------------------------------------------------------------------------------------------------------------

# Step 2: Define the UNet model
def create_unet_model(input_shape, output_shape=(128, 128, 16), output_channels=1):
    model = Sequential()

    # Encoder
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # Decoder
    model.add(Conv3DTranspose(64, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(Conv3DTranspose(output_channels, kernel_size=(3, 3, 3), activation='sigmoid', padding='same'))

    # Add a Flatten layer to convert 3D output to 1D
    model.add(Flatten())

    # Calculate the total number of elements in the output shape
    output_size = np.prod(output_shape)

    # Add a Dense layer with the correct number of units
    model.add(Dense(units=output_size, activation='sigmoid'))

    # Add a Reshape layer with the correct target shape
    model.add(Reshape(target_shape=output_shape))

    return model

# Step 3: Compile the model
model = create_unet_model(input_shape=(512, 512, 58, 1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Step 5: Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Step 6: Make predictions on the test set
predictions = model.predict(X_test)

# Step 7: Visualization (Optional)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

