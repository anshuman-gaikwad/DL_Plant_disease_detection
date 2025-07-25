# Plant Disease Detection with ResNet50

This deep learning project uses a fine-tuned ResNet50 model to classify plant diseases from images. Built with TensorFlow and Keras, it incorporates data augmentation and a Gradio interface for interactive predictions.

## Project Overview
The project leverages ResNet50, pre-trained on ImageNet, to detect plant diseases from a custom dataset (`tree_data1`). The model is fine-tuned with additional layers and data augmentation to enhance accuracy. A Gradio interface allows users to upload plant images for real-time disease classification.

### Key Features
- **Model**: ResNet50 with custom layers (GlobalAveragePooling2D, Dense, Dropout).
- **Dataset**: Images in `tree_data1/` organized by class (e.g., `train`, `test`).
- **Input**: 224x224 RGB images.
- **Output**: Disease classification with confidence scores.
- **Interface**: Gradio app for user-friendly predictions.

## Usage
1. Run the notebook:
   - Open `notebooks/LAB04 (1).ipynb` in Jupyter Notebook or Google Colab.
   - For Colab, mount Google Drive to access `tree_data1/`.
   - Execute cells to train the model and launch the Gradio interface.
2. Use the Gradio interface:
   - Upload a plant image to get disease predictions and confidence scores.

## Model Details
- **Architecture**: ResNet50 (pre-trained) with:
  - GlobalAveragePooling2D
  - Dense (1024 units, ReLU)
  - Dropout (0.5)
  - Dense (3 units, softmax for 3 classes)
- **Training**:
  - Optimizer: Adam (learning rate = 0.001)
  - Loss: Categorical Crossentropy
  - Metrics: Accuracy
  - Augmentation: Rotation, zoom, flip, shear
- **Performance**: Validation loss: 0.7280, Validation accuracy: 0.7643
- **Output**: Model saved as `my_model.keras` (not included in repository).

## File Structure
- `notebooks/LAB04 (1).ipynb`: Main notebook for model training, evaluation, and Gradio interface.
- `requirements.txt`: Lists Python dependencies for the project.

## Notes
- **Colab Compatibility**: The notebook uses Google Drive for dataset access. For local use, modify `dataset_path` to point to your local `tree_data1/` directory.
- **Dataset**: Not included in the repository due to size; users must provide their own `tree_data1/` with class subfolders (`train`, `test`, etc.).
- **Model Weights**: The trained model (`my_model.keras`) is not included; users must train the model using the notebook.
- **Class Names**: The dataset contains 3 classes (`test`, `train`, `tree_data`), which may need adjustment for meaningful disease labels.

## Contributing
To contribute:
1. Fork the repository.
2. Create a branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.