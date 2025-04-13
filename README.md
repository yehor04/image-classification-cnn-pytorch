# image-classification-cnn-pytorch

A PyTorch-based convolutional neural network for 20-class grayscale image classification.  
This project was developed as part of the *Programming in Python II* course at **JKU Linz**.

---

## Description

This repository contains a complete implementation of a CNN model from scratch (no transfer learning), trained on a dataset of 12,000+ pre-labeled grayscale images (resized to 100x100 px).  
The model classifies images into 20 object categories such as:

book, bottle, car, cat, chair, computermouse, cup, dog, flower, fork, glass, glasses, headphones, knife, laptop, pen, plate, shoes, spoon, tree;

## Project Structure

├── architecture.py # Contains the MyCNN class ├── training.py # Training loop with loss, optimizer, scheduler ├── dataset.py # ImageDataset class with preprocessing pipeline ├─ README.md # Project description

Final test accuracy: 75,44 %
Model evaluated using a hidden test set via the official JKU Challenge Server.
