# Importing all the necessary libraries

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from skimage import exposure, filters

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

#defining unet 3d model
def unet_3d_model(input_shape=(512, 512, 58, 1)):
    inputs = layers.Input(input_shape)

    # Encoder
    conv1 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    # Mid-level
    conv3 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)

    # Decoder
    up1 = layers.Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv3)
    concat1 = layers.Concatenate(axis=-1)([conv2, up1])
    conv4 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(concat1)
    conv4 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)

    up2 = layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4)
    concat2 = layers.Concatenate(axis=-1)([conv1, up2])
    conv5 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(concat2)
    conv5 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv5)

    # Output layer
    output = layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(conv5)

    model = models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Create the 3D U-Net model
model_3d = unet_3d_model()

# Print the model summary
model_3d.summary()
