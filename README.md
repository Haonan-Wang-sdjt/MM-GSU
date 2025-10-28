# MM-GSU
**Multi-Modal Gated Semantic Understanding Network**

## Project Overview

MM-GSU (Multi-Modal Gated Semantic Understanding Network) is a multi-modal learning project that combines traffic parameters and image feature data for analysis and prediction using deep learning models. The project aims to improve traffic management systems and provides a new approach for the joint analysis of image and traffic flow data.

## File Structure

MM-GSU/
├── datasets/             # Contains traffic parameter and image feature datasets  
│   ├── traffic/          # Traffic parameter dataset with 30 folders, each containing multiple traffic data files  
│   └── images/           # Image feature dataset with 30 folders, each containing multiple image data files  
├── images/               # Stores images from the paper (e.g., charts, experimental results)  
├── src/                  # Contains code for model training and prediction  
│   └── MM-GSU.py         # Main script for model training and prediction  
├── .gitignore            # Git ignore file  
├── LICENSE               # License file  
├── README.md             # Project description document  

## Datasets

### Traffic Parameter Dataset (`datasets/traffic/`)

This dataset contains 30 folders, each representing a set of data. Each folder includes multiple traffic parameter data files, which are used for traffic flow analysis and prediction tasks. The data format and content can be adjusted according to specific requirements.

### Image Feature Dataset (`datasets/images/`)

This dataset also contains 30 folders, which correspond to the folders of the traffic parameter dataset. Each folder represents a distinct set of image data and includes multiple image features. These image feature data can be further analyzed using deep learning models.

## src

The `src` directory contains the code responsible for training and predicting using the **Multi-Modal Gated Semantic Understanding Network (MM-GSU)**. The main script `MM-GSU.py` is designed to be easily adaptable for different datasets by modifying a few key paths.

The main steps in the `MM-GSU.py` script include:

1. **Data Preparation**: Loading traffic parameter and image feature data from specified directories.
2. **Model Definition**: Defining the architecture of the Multi-Modal Gated Semantic Understanding Network.
3. **Training**: The model is trained using the provided datasets, with options to adjust training parameters such as batch size and learning rate.
4. **Prediction**: Once trained, the model can make predictions on unseen data.
5. **Model Saving**: After training, the model weights are saved for future use.

### Customizing the Script

To run the script, you only need to adjust a few paths in the code:

- **Traffic Data Path**: Change the `traffic_npy_dir` variable to point to your local traffic parameter data folder.
- **Image Data Path**: Modify the `image_npy_dir` variable to point to your local image feature data folder.
- **Output Path**: Optionally, set the `output_dir` variable to specify where you want to save your predictions.

Example:

```python
# =======================
# ✅ Main program entry
# =======================
if __name__ == '__main__':
    train = True  # Set to True for retraining
    
    # Change these paths to match your local directories
    traffic_npy_dir = r"C:\Users\czx43\Desktop\MM-GSU\datasets\traffic_npy"
    image_npy_dir = r"C:\Users\czx43\Desktop\MM-GSU\datasets\image_npy"
    
    output_dir = "prediction_200"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if train:
        # Train the model
        train_model(traffic_npy_dir, image_npy_dir)
```
## Installation

### Clone the Repository

You can clone the repository to your local machine using the following command:

```bash
git clone https://github.com/Haonan-Wang-sdjt/MM-GSU.git
```
## Install Dependencies

Make sure you have Python installed (recommended version 3.x). You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```
## Run the Script
1. Navigate to the project directory:

```bash
cd MM-GSU
```
2. Set the correct paths for your datasets in the `MM-GSU.py` script, specifically the paths for the `traffic_npy_dir` and `image_npy_dir` variables.

   Example:
```python
traffic_npy_dir = r"C:\path\to\your\traffic_npy"
image_npy_dir = r"C:\path\to\your\image_npy"
```
3. If everything is set up correctly, you can run the script to train the model:
```bash
python src/MM-GSU.py
```
4. The model will start training if the `train` variable is set to `True`. After training, the model can be used for predictions on new data.

   Note: You can modify the script to switch to prediction mode by setting the `train` variable to `False`.

## Output Directory
You can specify where the trained model’s predictions will be saved by modifying the `output_dir` variable. For example:

```python
output_dir = "prediction_output"
```
This will save your predictions in the `prediction_output` folder.
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
