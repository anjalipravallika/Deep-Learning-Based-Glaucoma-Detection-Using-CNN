This study develops a deep learning model to detect glaucoma using images of the eye. The
model was trained with images focused on specific areas of the eye (regions of interest, or
ROIs) and used data augmentation techniques, such as flipping the images, to improve
performance.
The model has multiple layers that analyze the images, ending with a decision layer that
classifies the images as either showing glaucoma or not. The implementation was tested on
both CPU and GPU environments, and the time taken for training and inference was compared.
The GPU significantly reduced the time required, making it more suitable for handling large
datasets.
The model was trained for 30 epochs using the Adam optimizer and a special loss function
called Focal Loss to help deal with imbalanced data. After training, the model showed results
with a low loss value (0.0592) and accuracy (97.95%) on the training data. When tested on
new data, it achieved a precision of 98.10%, recall of 96.88%, and an overall accuracy of
98.05%. These results demonstrate that the model can accurately detect glaucoma, making it a
useful tool for automatic detection in medical imaging. 
