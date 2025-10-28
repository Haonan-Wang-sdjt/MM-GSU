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

This dataset contains 30 folders, each representing a traffic area. Each folder contains multiple traffic parameter data files, which are used for traffic flow analysis and prediction tasks. The data format and content can be adjusted based on specific needs.

### Image Feature Dataset (`datasets/images/`)

This dataset also contains 30 folders, each representing a different set of image data. Each folder contains multiple image files that are used for image feature extraction and classification tasks. These image data can be further analyzed using deep learning models.

## Installation

### Clone the Repository

You can clone the repository to your local machine using the following command:

```bash
git clone https://github.com/Haonan-Wang-sdjt/MM-GSU.git
