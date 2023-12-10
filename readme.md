# Hypodensity Detection in Brain NCCT Images

## Background

This project aims to automate and streamline the identification of early ischemic changes in acute stroke patients through the segmentation of hypodense regions in Brain Non-Contrast Computed Tomography (NCCT) images. Early ischemic changes are visualized as hypoattenuation of brain tissue with a loss of grey-white matter differentiation. The goal is to develop a robust and efficient algorithm or AI model that can accurately segment hypodense regions, regardless of slice thickness and image orientation.

## Project Overview

### Libraries Used

- Python 3.11
- TensorFlow
- NumPy
- nibabel
- scikit-learn
- matplotlib

### Steps

1. **Dataset Loading:**
   - Load the dataset containing 3D axial thick slice NCCT images and their corresponding hypodensity region masks in NIFTI format.
   - Each image pair is grouped in folders named after their 'Case ID.'

2. **Data Preprocessing:**
   - Normalize pixel values in the NCCT images to improve the signal-to-noise ratio.
   - Additional preprocessing steps may include bias field correction, intensity normalization, and skull stripping.

3. **Data Preparation:**
   - Split the dataset into training and validation sets.
   - Create 3D U-Net model input and output data.

4. **3D U-Net Model Definition:**
   - Build a 3D U-Net model using TensorFlow and Keras.
   - Use Conv3D and Conv3DTranspose layers for encoding and decoding.

5. **Model Training:**
   - Train the 3D U-Net model on the prepared dataset.
   - Optimize using the Adam optimizer and binary crossentropy loss.

6. **Model Evaluation:**
   - Evaluate the model using metrics such as accuracy and loss on the validation set.
   - Use additional metrics like Dice score for hypodensity segmentation.

7. **Visualization:**
   - Visualize sample slices of NCCT images and corresponding hypodensity masks for qualitative assessment.

## Pseudocode

```python
# Pseudocode for Hypodensity Detection Algorithm

# Load Dataset
data_dir = "path/to/your/dataset"
case_ids = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

# Preprocessing
for case_id in case_ids:
    ncct_image_path = os.path.join(data_dir, case_id, f"{case_id}_NCCT.nii.gz")
    roi_mask_path = os.path.join(data_dir, case_id, f"{case_id}_ROI.nii.gz")

    # Load NCCT image and ROI mask using nibabel
    ncct_image = nib.load(ncct_image_path).get_fdata()
    roi_mask = nib.load(roi_mask_path).get_fdata()

    # Apply preprocessing (e.g., intensity normalization)

# Data Preparation
X_train, X_val, y_train, y_val = train_test_split(...)
# Further processing and expansion of dimensions for model input

# Model Definition
model_3d = unet_3d_model(input_shape=(512, 512, 58, 1))

# Model Training
history = model_3d.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Model Evaluation
val_loss, val_accuracy = model_3d.evaluate(X_val, y_val, verbose=1)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")

# Visualization
# Sample code to visualize NCCT images and hypodensity masks
plt.imshow(ncct_image[:, :, slice_index], cmap='gray')
plt.imshow(roi_mask[:, :, slice_index], cmap='viridis')
plt.show()
