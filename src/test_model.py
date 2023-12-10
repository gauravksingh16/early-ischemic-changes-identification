# Importing necessary libraries for testing
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score

from src.detection import X_test, y_test


def test_model(model_path, X_test, y_test):
    # Load the pre-trained model
    pretrained_model = load_model(model_path)

    # Test the model on the test set
    test_results = pretrained_model.evaluate(X_test, y_test)

    # Display test results
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1] * 100:.2f}%")

    # Make predictions on the test set
    predictions = pretrained_model.predict(X_test)

    # Additional: Evaluate additional metrics like accuracy, F1 score, etc.
    # You may need to threshold predictions for binary segmentation tasks
    binary_predictions = (predictions > 0.5).astype(int)

    # Example: Calculate accuracy and F1 score
    accuracy = accuracy_score(y_test.flatten(), binary_predictions.flatten())
    f1 = f1_score(y_test.flatten(), binary_predictions.flatten())

    print(f"Test Accuracy (Binary): {accuracy * 100:.2f}%")
    print(f"Test F1 Score (Binary): {f1:.4f}")

if __name__ == "__main__":
    # Provide the path to your pre-trained model
    model_path = 'src/hypodensity_segmentation_model.h5'

    # Assuming you already have X_test and y_test from your dataset
    # Replace this with your actual data loading code
    # X_test, y_test = load_test_data()

    # Call the test_model function
    test_model(model_path, X_test, y_test)