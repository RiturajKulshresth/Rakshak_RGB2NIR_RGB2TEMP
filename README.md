# Mapping-RGB-face-video-data-to-core-body-temperature-at-runtime-RAKSHAK

The objective is mapping the following -
1. RGB image <-> Body Temperature
2. NIR image <-> Body Temperature, but this requires RGB -> NIR conversion

Therfore, this project is divided into 2 parts -
1. To find a relation between the temperature of a human body and the image taken by a camera. This is to be done using both RGB and NIR images of the face of a person
2. The second part of the project is where we try to generate the NIR image from RGB images of the human faces.

## Datasets

### Dataset I

The first data was collected by taking RGB images of the forehead regions of the participants and taking their core body temperature using an IR thermometer/thermal gun in a room with artificial lighting. The images correspond to the temperatures 96°F, 97°F, 98°F, 99°F, and 100°F. The temperature was measured for varying times of the day after various activities which were also included. However, due to the COVID restrictions, the data collection was limited to a few participants causing the data to be skewed towards the normal body temperature of about 97 / 98 °F which caused problems in training the model with the data.

### EPFL Dataset

This dataset contains a set of RGB and NIR images of different scenery ranging from forests and lakes to cities and buildings. The dataset consisted of 477 pairs of RGB and NIR images. The images were captured using separate exposures from modified SLR cameras, using visible and NIR filters. The cutoff wavelength between the two filters for RGB and NIR images is approximately 750nm.

### IITJ Dataset

The data was collected on the IITJ campus, which includes RGB and thermal videos of participants using a thermal camera, their core body temperature, and factors that can affect the data such as ambient temperature, activity before collection. This data had temperatures in the 96°F, 97°F, and 98°F range. The data presently being used still have posed the same problem that we encountered before and data from more participants on different conditions is necessary to balance the dataset.

## RGB <-> Temperature mapping (using Dataset I)

To find any relation between the RGB image of a person and the temperature of the person we have used convolutional neural networks. The images have been divided based on their respective body temperature and various convolutional neural networks were experimented with to find any underlying mapping function. The data was resized to 256 x 256 pixels for easy training. Appropriate weights were added to each temperature class in order to adjust for the unequal distribution of data.

We experimented with different models with results in the accuracy range of 20 % to 50 % and losses in the range of 2 to 4.

## Generation of NIR images from RGB (using EPFL dataset)

The data is pre-processed and augmented for training. We tried to solve this problem using different approaches with multiple models in each approach. Finally, the best result was obtained and after some post-processing we get the below result -

<p align="center">
<img src="https://github.com/saurabhburewar/Mapping-RGB-face-video-data-to-core-body-temperature-at-runtime-RAKSHAK/blob/main/Results/Original%20RGB%20rescaled.png"><img src="https://github.com/saurabhburewar/Mapping-RGB-face-video-data-to-core-body-temperature-at-runtime-RAKSHAK/blob/main/Results/original%20NIR%20rescaled.png"><img src="https://github.com/saurabhburewar/Mapping-RGB-face-video-data-to-core-body-temperature-at-runtime-RAKSHAK/blob/main/Results/contrast%20adjusted.png">
</p>
<p align="center">
RGB Image - Ground Truth - Our Result
</p>

## Generation of NIR images from RGB (using IITJ dataset)

Similar experiments with the IITJ datasets gave us the below results -

<p align="center">
<img src="https://github.com/saurabhburewar/Mapping-RGB-face-video-data-to-core-body-temperature-at-runtime-RAKSHAK/blob/main/Results/0.jpg"><img src="https://github.com/saurabhburewar/Mapping-RGB-face-video-data-to-core-body-temperature-at-runtime-RAKSHAK/blob/main/Results/92_0_256x7_100.png">
</p>

## Code

All the code files are organized according to the datasets that it operates on. Some of them are experiments done along the way to achieve the final results and the rest are the actual code that contribute to the final result. To run and get the resluts shown above, follow the steps given below 

### Run for Dataset I 
All files can be found in the directory "Final_codes/Dataset_I". To get the temperature prediction using selfdata run in the following order

- "Final codes/Dataset I/forehead crop.ipynb" to crop the original data
- "Final codes/Dataset I/Dataset_generation_selfdata.ipynb" to generate dataset
- "Final codes/Dataset I/Selfdata_model.ipynb" to get the models
- "Final codes/Dataset I/Predict_selfdata_temp.ipynb" to predict the temperature of the forehead

### Run for EPFL Dataset


### Run for IITJ Dataset
All files can be found in directory "Final_codes/IITJ_Dataset". To get the RGB to NIR and temperature prediction using IITJ dataset, run in the following order

- "VidToImg.py" to convert the video data to frames and extract the faces
- "RGB_to_NIR_Rakshak Colab 1.ipynb" to trian the model and predict NIR equivalent of an RGB image
- "RGBtoTEMP_Rakshak.ipynb" to predict temperatures
