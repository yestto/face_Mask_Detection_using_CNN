 Face Mask Detection using CNN
This project implements a Convolutional Neural Network (CNN) to detect whether individuals in images are wearing a face mask or not. It uses a binary classification model trained on a labeled dataset of facial images.

📁 Project Structure
Copy
Edit
DL_Project_5_face_Mask_Detection_using_CNN.ipynb
README.md
📦 Features
Preprocesses and loads a dataset of face images with/without masks.

Builds a custom CNN from scratch using TensorFlow/Keras.

Applies data augmentation to improve generalization.

Evaluates performance with accuracy, classification report, and confusion matrix.

Visualizes training metrics and predictions.

🧠 Model Architecture
Custom CNN consisting of:

Multiple Convolutional layers with ReLU activation

MaxPooling layers for downsampling

Dropout layers to prevent overfitting

Fully Connected (Dense) layers

Final output layer with 1 neuron + sigmoid for binary classification

🧪 Training Details
Optimizer: Adam

Loss Function: Binary Crossentropy

Epochs: 10–20

Batch Size: 32

Includes callbacks: EarlyStopping, ModelCheckpoint

🧾 Dataset
Categories: With Mask, Without Mask

Images are resized to 100x100 pixels (adjustable)

Split into training, validation, and test sets

Directory-based dataset loading using ImageDataGenerator.flow_from_directory

📊 Evaluation Metrics
Accuracy

Precision, Recall, F1-score via classification report

Confusion Matrix

Loss/accuracy visualization during training

🖼️ Sample Output
Notebook includes:

Plots of model accuracy and loss

Random predictions on test images with labels

🚀 Getting Started
Prerequisites
Python ≥ 3.7

TensorFlow ≥ 2.x

NumPy, Matplotlib, scikit-learn

Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yestto/FaceMaskDetection-CNN.git
cd FaceMaskDetection-CNN
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Running the Notebook
Launch the notebook:

bash
Copy
Edit
jupyter notebook DL_Project_5_face_Mask_Detection_using_CNN.ipynb
Make sure your dataset folder is properly structured like:

bash
Copy
Edit
dataset/
├── train/
│   ├── with_mask/
│   └── without_mask/
├── val/
│   ├── with_mask/
│   └── without_mask/
├── test/
│   ├── with_mask/
│   └── without_mask/
📈 Example Accuracy
Training Accuracy: ~97%

Validation Accuracy: ~95%

Test Accuracy: ~94%
