# Predicting Stock Prices Using LSTM-CNN-CBAM Model

# DS-UA 301 Final Project
Members: Zirui(Frank) Feng, Zongjian(James) Li

# Project Description
The goal of this project is to develop a prediction model, Long Short-Term Memory (LSTM) embedded with Convolutional Neural Network (CNN), to predict stock price (the highest price of the day) movements based on data from Kaggle NYSE dataset. We insert Convolutional Block Attention Module (CBAM) to the base model and compare model performances of LSTM-CNN with four different attention modules: Efficient Chanel Attention (ECA), Chanel Attention Module (CAM), Spatial Attention Module (SAM), and Convolutional Block Attention Module (CBAM) to demonstrate that inserting CBAM to LSTM-CNN is effective. 

# Repository Description and Code Structures
Datasets: Dataset used in the project, and features including code, date, open, low, close, high

Dataloder.py: Prepare datasets into the test/train splitting form based on the input features

model.py: Construct models including LSTM model, LSTM with ECA model, LSTM with CAM model, LSTM with SAM model and LSTM with CBAM model

train.py: Train models based on dataset

test.py: Test models, report RMSE and draw prediction graphs of four models based on dataset

DS-UA 301 Final Project: Presentation slide

result_picture: File including graphs of four models

best_model: File including best trained models


# Example Commands



# Results

# Conclusion

