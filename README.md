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
# Training Process
```
python train.py -m Base

python train.py -m ECA

python train.py -m CAM

python train.py -m SAM

python train.py -m CBAM
```
Models will be stored in best_model folder


# Testing Process
```
python test.py -m Base

python test.py -m ECA

python test.py -m CAM

python test.py -m SAM

python test.py -m CBAM
```
Prediction results will be stored in result_picture folder


# Results

| Model  | RMSE |
| ------------- | ------------- |
| CNN+LSTM  | 0.00011371035229372369  |
| CNN+LSTM+ECA  | 0.0001245921911587092 |
| CNN+LSTM+CAM  | 0.00009550479312152179  |
| CNN+LSTM+SAM  | 0.00041322291971565306  |
| CNN+LSTM+CBAM  | 0.0003162174993617968 |

Base Model
![Base_fic](https://user-images.githubusercontent.com/94018723/208316303-dfb10e2d-aa18-4472-9ad7-0d2ba4ab6028.jpg)

Base Model + ECA
![ECA_fic](https://user-images.githubusercontent.com/94018723/208316332-c4b2b94c-5b40-4c0a-afe5-eeb82f6c2a80.jpg)

Base Model + CAM
![SE_fic](https://user-images.githubusercontent.com/94018723/208316360-8dec44a1-d4ea-435b-b364-33810a68fc10.jpg)

Base Model + SAM
![HW_fic](https://user-images.githubusercontent.com/94018723/208316374-f4b60fce-8d38-4207-9a4e-8939f854c910.jpg)

Base Model + CBAM
![CBAM_fic](https://user-images.githubusercontent.com/94018723/208316388-9e3f2576-19ae-4eb7-8e34-b50c9c329d38.jpg)


# Conclusion
Comparing the RMSE of four models, CNN-LSTM with embedded Channel Attention Module has the lowest RMSE (0.00009550479312152179). Although we aim to use the CNN-LSTM-CBMA model to learn the fluctuation of stock price to achieve a better prediction performance, the model is not really sensitive to price fluctuations. This phenomenon is might due to the fact that CBAM is more suitable for image classification field.
