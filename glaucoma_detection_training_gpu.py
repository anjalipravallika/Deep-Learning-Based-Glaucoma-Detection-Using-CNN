

import os
os.environ["KERAS_BACKEND"] = "jax" # you can also use tensorflow or torch

import cv2
import keras
#from keras import ops
import tensorflow as tf

#import cv2|
import pandas as pd
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
import joblib

import matplotlib.pyplot as plt




import zipfile
import os
# Check if TensorFlow can access the GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("Num GPUs Available: ", len(physical_devices))
else:
    print("No GPUs found. Using CPU.")

# Load the trained model inside a GPU context
with tf.device('/GPU:0'):

    # Directory containing the images
    train_dir = "C:/Users/prane/Downloads/Glaucoma_Detection/fundus/train"

    # List to store the file paths and labels
    data = []

    # Iterate over the subdirectories 0 and 1
    for label, subdir in enumerate(["0", "1"]):
        # Get the full path of the current subdirectory
        subdir_path = os.path.join(train_dir, subdir)
        # Iterate over the files in the subdirectory
        for filename in os.listdir(subdir_path):
            # Check if the file has a .png extension
            if filename.endswith(".png"):
                # Create the full path of the file
                file_path = os.path.join(subdir_path, filename)
                # Append the file path and label to the list
                data.append((file_path, label))

    # Create a DataFrame from the list of file paths and labels
    df_train = pd.DataFrame(data, columns=["file_path", "label"])

    # Save the DataFrame to a CSV file
    df_train.to_csv("train.csv", index=False)

    #####################################################################################################
    # Directory containing the images
    test_dir = "C:/Users/prane/Downloads/Glaucoma_Detection/fundus/test"

    # List to store the file paths and labels
    data = []

    # Iterate over the subdirectories 0 and 1
    for label, subdir in enumerate(["0", "1"]):
        # Get the full path of the current subdirectory
        subdir_path = os.path.join(test_dir, subdir)
        # Iterate over the files in the subdirectory
        for filename in os.listdir(subdir_path):
            # Check if the file has a .png extension
            if filename.endswith(".png"):
                # Create the full path of the file
                file_path = os.path.join(subdir_path, filename)
                # Append the file path and label to the list
                data.append((file_path, label))

    # Create a DataFrame from the list of file paths and labels
    df_test = pd.DataFrame(data, columns=["file_path", "label"])

    # Save the DataFrame to a CSV file
    df_test.to_csv("test.csv", index=False)
    ##########################################################################################

    # Directory containing the images
    valid_dir = "C:/Users/prane/Downloads/Glaucoma_Detection/fundus/val"

    # List to store the file paths and labels
    data = []

    # Iterate over the subdirectories 0 and 1
    for label, subdir in enumerate(["0", "1"]):
        # Get the full path of the current subdirectory
        subdir_path = os.path.join(valid_dir, subdir)
        # Iterate over the files in the subdirectory
        for filename in os.listdir(subdir_path):
            # Check if the file has a .png extension
            if filename.endswith(".png"):
                # Create the full path of the file
                file_path = os.path.join(subdir_path, filename)
                # Append the file path and label to the list
                data.append((file_path, label))

    # Create a DataFrame from the list of file paths and labels
    df_valid = pd.DataFrame(data, columns=["file_path", "label"])

    # Save the DataFrame to a CSV file
    df_valid.to_csv("valid.csv", index=False)

    # Load the CSV file into a DataFrame
    df = pd.read_csv("C:/Users/prane/Downloads/Glaucoma_Detection/G1020_final/G1020.csv")

    # Create a new DataFrame with the image names and labels
    df_train1 = pd.DataFrame({
        'file_path': ["C:/Users/prane/Downloads/Glaucoma_Detection/G1020_final/Images/" + image_id for image_id in df['imageID']],
        'label': df['binaryLabels']
    })

    # Display the first few rows of the new DataFrame
    print(df_train1.head())

    df_train1.head(1000)

    # Load the CSV file into a DataFrame
    df = pd.read_csv("C:/Users/prane/Downloads/Glaucoma_Detection/ORIGA/OrigaList.csv")

    # Create a new DataFrame with the image names and labels
    df_train2 = pd.DataFrame({
        'file_path': ["C:/Users/prane/Downloads/Glaucoma_Detection/ORIGA/Images/" + image_id for image_id in df['Filename']],
        'label': df['Glaucoma']
    })

    # Display the first few rows of the new DataFrame
    print(df_train2.head())

    # Load the CSV file into a DataFrame
    df = pd.read_csv("C:/Users/prane/Downloads/Glaucoma_Detection/ORIGA/OrigaList.csv")

    # Create a new DataFrame with the image names and labels
    df_train2 = pd.DataFrame({
        'file_path': ["C:/Users/prane/Downloads/Glaucoma_Detection/ORIGA/Images/" + image_id for image_id in df['Filename']],
        'label': df['Glaucoma']
    })

    # Display the first few rows of the new DataFrame
    print(df_train2.head())

    # Directory containing the images
    train_dir = "C:/Users/prane/Downloads/Glaucoma_Detection/OCT/dataset/dataset"

    # List to store the file paths and labels
    data = []

    # Iterate over the subdirectories 0 and 1
    for label, subdir in enumerate(["yes"], start=1):  # Start from 1
        # Get the full path of the current subdirectory
        subdir_path = os.path.join(train_dir, subdir)
        # Iterate over the files in the subdirectory
        for filename in os.listdir(subdir_path):
            # Check if the file has a .png extension
            if filename.endswith(".jpg"):
                # Create the full path of the file
                file_path = os.path.join(subdir_path, filename)
                # Append the file path and label to the list
                data.append((file_path, label))

    # Create a DataFrame from the list of file paths and labels
    df_train3 = pd.DataFrame(data, columns=["file_path", "label"])

    # Save the DataFrame to a CSV file
    df_train3.to_csv("train1.csv", index=False)

    # Directory containing the images
    train_dir = "C:/Users/prane/Downloads/Glaucoma_Detection/G1020/G1020"

    # List to store the file paths and labels
    data = []

    # Iterate over the subdirectories 0 and 1
    for label, subdir in enumerate(["glucoma"], start=1):  # Start from 1
        # Get the full path of the current subdirectory
        subdir_path = os.path.join(train_dir, subdir)
        # Iterate over the files in the subdirectory
        for filename in os.listdir(subdir_path):
            # Check if the file has a .png extension
            if filename.endswith(".jpg"):
                # Create the full path of the file
                file_path = os.path.join(subdir_path, filename)
                # Append the file path and label to the list
                data.append((file_path, label))

    # Create a DataFrame from the list of file paths and labels
    df_train4 = pd.DataFrame(data, columns=["file_path", "label"])

    # Save the DataFrame to a CSV file
    df_train4.to_csv("train2.csv", index=False)

    # Directory containing the images
    train_dir = "C:/Users/prane/Downloads/Glaucoma_Detection/EYEPACS/release-raw/release-raw/train"

    # List to store the file paths and labels
    data = []

    # Iterate over the subdirectories 0 and 1
    for label, subdir in enumerate(["RG"], start=1):  # Start from 1
        # Get the full path of the current subdirectory
        subdir_path = os.path.join(train_dir, subdir)
        # Iterate over the files in the subdirectory
        for filename in os.listdir(subdir_path):
            # Check if the file has a .png extension
            if filename.endswith(".jpg"):
                # Create the full path of the file
                file_path = os.path.join(subdir_path, filename)
                # Append the file path and label to the list
                data.append((file_path, label))

    # Create a DataFrame from the list of file paths and labels
    df_train5 = pd.DataFrame(data, columns=["file_path", "label"])

    # Save the DataFrame to a CSV file
    df_train5.to_csv("train3.csv", index=False)

    df_train = pd.concat([df_train, df_train1, df_train2, df_train3, df_train4, df_train5], ignore_index=True)

    # Display the first few rows of the new DataFrame
    print(df_train.head())

    df_train.label.value_counts()

    df_train.head()

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # Load the image
    image_path = df_train1["file_path"].sample().values[0]
    #image_path = os.path.join(file_path, file_name)
    print(f"Attempt:  {image_path}")


# Load the image
    image = cv2.imread(image_path)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the image to grayscale
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (65, 65), 0)

    # Find the pixel with the highest intensity value
    max_intensity_pixel = np.unravel_index(np.argmax(blurred_image), blurred_image.shape)

    # Define the radius for the circle
    radius = 200 // 2

    # Get the x and y coordinates for cropping the image
    x = max_intensity_pixel[1] - radius
    y = max_intensity_pixel[0] - radius

    # Create a mask for the circle
    mask = np.zeros_like(image)
    cv2.circle(mask, (x + radius, y + radius), radius, (255, 255, 255), -1)

    # Apply the mask to the original image
    roi_image = cv2.bitwise_and(image, mask)

    # Plot the original image and the extracted ROI
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
    plt.title('Extracted ROI')
    plt.axis('off')

    plt.show()

    # Load the image
    image_path = df_train["file_path"].sample().values[0]

    # Check the file extension
    file_extension = os.path.splitext(image_path)[1]

    # Load the image based on the file extension
    if file_extension.lower() == '.png':
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    else:
        image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (65, 65), 0)

    # Find the pixel with the highest intensity value
    max_intensity_pixel = np.unravel_index(np.argmax(blurred_image), blurred_image.shape)

    # Define the radius for the circle
    radius = 200 // 2

    # Get the x and y coordinates for cropping the image
    x = max_intensity_pixel[1] - radius
    y = max_intensity_pixel[0] - radius

    # Create a mask for the circle
    mask = np.zeros_like(image)
    cv2.circle(mask, (x + radius, y + radius), radius, (255, 255, 255), -1)

    # Apply the mask to the original image
    roi_image = cv2.bitwise_and(image, mask)

    # Plot the original image and the extracted ROI
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
    plt.title('Extracted ROI')
    plt.axis('off')

    plt.show()

    def extract_rois_and_labels(df, output_dir):
        with tf.device('/GPU:0'):
        # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Initialize an empty list to store the rows of the DataFrame
            roi_rows = []

            for index, row in df.iterrows():
                # Load the image
                image = cv2.imread(row['file_path'])

                # Resize the image to 512x512
                resized_image = cv2.resize(image, (512, 512))

                # Convert the resized image to grayscale
                gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

                # Apply Gaussian blur
                blurred_image = cv2.GaussianBlur(gray_image, (65, 65), 0)

                # Find the pixel with the highest intensity value
                max_intensity_pixel = np.unravel_index(np.argmax(blurred_image), blurred_image.shape)

                # Define the radius for the circle
                radius = 200 // 2

                # Get the x and y coordinates for cropping the image
                x = max_intensity_pixel[1] - radius
                y = max_intensity_pixel[0] - radius

                # Create a mask for the circle
                mask = np.zeros_like(resized_image)
                cv2.circle(mask, (x + radius, y + radius), radius, (255, 255, 255), -1)

                # Apply the mask to the resized image
                roi_image = cv2.bitwise_and(resized_image, mask)

                # Split the green channel
                green_channel = roi_image[:, :, 1]
                # Apply histogram equalization
                clahe_op = cv2.createCLAHE(clipLimit=2)
                roi_image = clahe_op.apply(green_channel)

                # Save the ROI image
                roi_filename = os.path.basename(row['file_path']).split('.')[0] + '_roi.jpg'
                roi_path = os.path.join(output_dir, roi_filename)
                cv2.imwrite(roi_path, roi_image)

                # Append the row to the list
                roi_rows.append({'roi_path': roi_path, 'label': row['label']})

            # Create a DataFrame from the list of rows
            roi_df = pd.DataFrame(roi_rows)

            return roi_df

    # Extract ROIs and labels for the training images
    train_roi_df = extract_rois_and_labels(df_train, 'train_roi_images')
    print("train roi extraion done")
    # Extract ROIs and labels for the testing images
    test_roi_df = extract_rois_and_labels(df_test, 'test_roi_images')
    print("test roi extraion done")

    # Extract ROIs and labels for the validation images
    valid_roi_df = extract_rois_and_labels(df_valid, 'valid_roi_images')
    print("val roi extraion done")

    def image_cvt_histeq(df, target_path):
        with tf.device('/GPU:0'):
            # Create the target directory if it doesn't exist
            if not os.path.exists(target_path):
                os.makedirs(target_path)

            new_df = pd.DataFrame(columns=['file_path', 'label'])

            for index, row in df.iterrows():
                # Load the image
                img = cv2.imread(row['file_path'])
                # Split the green channel
                green_channel = img[:, :, 1]
                # Apply histogram equalization
                clahe_op = cv2.createCLAHE(clipLimit=2)
                final_img = clahe_op.apply(green_channel)
                # Save the image
                cv2.imwrite(os.path.join(target_path, f'{index}.png'), final_img)
                # Add the new file path and label to the new DataFrame
                new_df = new_df.append({'file_path': os.path.join(target_path, f'{index}.png'), 'label': row['label']}, ignore_index=True)

            return new_df

    import random

    # Function to plot a random image from a DataFrame
    def plot_random_image(df, title):
        with tf.device('/GPU:0'):
            # Get a random row from the DataFrame-
            random_row = df.sample()

            # Load the image
            image_path = random_row['roi_path'].values[0]
            image = cv2.imread(image_path)

            # Convert the image from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Plot the image
            plt.imshow(image)
            plt.title(title)
            plt.axis('off')
            plt.show()

    # Plot a random image from each DataFrame
   # plot_random_image(train_roi_df, 'Training ROI')
   # plot_random_image(test_roi_df, 'Testing ROI')
   # plot_random_image(valid_roi_df, 'Validation ROI')

    train_roi_df.label.count()

    # Convert the label column to strings
    train_roi_df["label"] = train_roi_df["label"].astype(str)
    test_roi_df["label"] = test_roi_df["label"].astype(str)
    valid_roi_df["label"] = valid_roi_df["label"].astype(str)

    test_roi_df.label.value_counts()

    train_roi_df["label"].value_counts()

    def custom_preprocessing(image):
            with tf.device('/GPU:0'):
                # Convert the image to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Apply Gaussian blur
                blurred_image = cv2.GaussianBlur(gray_image, (65, 65), 0)

                # Find the pixel with the highest intensity value
                max_intensity_pixel = np.unravel_index(np.argmax(blurred_image), blurred_image.shape)

                # Define the radius for the circle
                radius = 200 // 2

                # Get the x and y coordinates for cropping the image
                x = max_intensity_pixel[1] - radius
                y = max_intensity_pixel[0] - radius

                # Create a mask for the circle
                mask = np.zeros_like(image)
                cv2.circle(mask, (x + radius, y + radius), radius, (255, 255, 255), -1)

                # Apply the mask to the original image
                roi_image = cv2.bitwise_and(image, mask)

                # Split the green channel
                green_channel = roi_image[:, :, 1]
                # Apply histogram equalization
                clahe_op = cv2.createCLAHE(clipLimit=2)
                roi_image = clahe_op.apply(green_channel)

                return roi_image

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Define the data generator for training images
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',

    )

    # Define the data generator for validation images
    valid_datagen = ImageDataGenerator(rescale=1./255)

    # Define the batch size
    batch_size = 8

    # Create the training data generator
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_roi_df,
        x_col='roi_path',
        y_col='label',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    # Create the validation data generator
    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid_roi_df,
        x_col='roi_path',
        y_col='label',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary',
    )

    # Define the data generator for test images
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Create the test data generator
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_roi_df,
        x_col='roi_path',
        y_col='label',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
    )

    import pandas as pd
    import numpy as np
    import cv2
    import tensorflow as tf  # Import TensorFlow
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.metrics import AUC, Precision, Recall  # Import Keras metrics
    from tensorflow.keras.losses import Loss
    from tensorflow.keras.optimizers import Adam  # Import Adam optimizer

    # Define the model
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        # Dropout(0.5),  # Uncomment this line if you want to add dropout
        Dense(1, activation='sigmoid')
    ])

    # Optimizer
    sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    # Define the FocalLoss class
    class FocalLoss(Loss):
        def __init__(self, alpha=0.65, gamma=2.0):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma

        def call(self, y_true, y_pred):
            # Calculate binary cross-entropy loss
            bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)

            # Calculate focal loss
            pt = tf.exp(-bce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

            # Calculate class weights
            total_samples = tf.reduce_sum(y_true)
            class_weights = total_samples / (2.0 * tf.reduce_sum(y_true, axis=0))

            # Apply class weights to the focal loss
            focal_loss = focal_loss * class_weights

            return focal_loss

    # Compile the model
    model.compile(optimizer=Adam(1e-4), loss=FocalLoss(), metrics=['binary_accuracy', AUC(), Precision(), Recall()])

    # Print model summary
    model.summary()

    # import os
    # import numpy as np
    # import pandas as pd
    # import tensorflow as tf
    # from tensorflow.keras.preprocessing.image import ImageDataGenerator
    # from tensorflow.keras.models import Sequential
    # from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    # from tensorflow.keras.metrics import AUC, Precision, Recall
    # from tensorflow.keras.optimizers import Adam
    # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    # # Assuming train_roi_df, valid_roi_df, and test_roi_df are already defined
    # # Define the batch size
    # batch_size =   8

    # # Create the data generators
    # train_datagen = ImageDataGenerator(
    #     rescale=1./255,
    #     horizontal_flip=True,
    #     vertical_flip=True,
    #     fill_mode='nearest'
    # )

    # valid_datagen = ImageDataGenerator(rescale=1./255)

    # # Create the training data generator
    # train_generator = train_datagen.flow_from_dataframe(
    #     dataframe=train_roi_df,
    #     x_col='roi_path',
    #     y_col='label',
    #     target_size=(256, 256),
    #     batch_size=batch_size,
    #     class_mode='binary',
    #     shuffle=True
    # )

    # # Create the validation data generator
    # valid_generator = valid_datagen.flow_from_dataframe(
    #     dataframe=valid_roi_df,
    #     x_col='roi_path',
    #     y_col='label',
    #     target_size=(256, 256),
    #     batch_size=batch_size,
    #     class_mode='binary'
    # )

    # # Define the model
    # model = Sequential([
    #     Conv2D(64, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    #     MaxPooling2D((2, 2)),
    #     Conv2D(128, (3, 3), activation='relu'),
    #     MaxPooling2D((2, 2)),
    #     Conv2D(128, (3, 3), activation='relu'),
    #     MaxPooling2D((2, 2)),
    #     Flatten(),
    #     Dense(512, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])

    # # Optimizer
    # optimizer = Adam(1e-4)

    # # Compile the model
    # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy', AUC(), Precision(), Recall()])

    # # Print model summary
    # model.summary()

    # # Define callbacks
    # early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-5)

    # # Calculate steps per epoch
    # steps_per_epoch = len(train_generator)  # Number of batches in the training data
    # validation_steps = len(valid_generator)  # Number of batches in the validation data

    # history = model.fit(
    #     train_generator,
    #     steps_per_epoch=int(steps_per_epoch),  # Ensure this is an integer
    #     epochs=2,
    #     validation_data=valid_generator,
    #     validation_steps=int(validation_steps),  # Ensure this is also an integer
    #     shuffle=True,
    #     callbacks=[early_stopping, reduce_lr]
    # )



    import os
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.metrics import AUC, Precision, Recall
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    # Assuming train_roi_df, valid_roi_df, and test_roi_df are already defined
    # Define the batch size
    batch_size = 8

    # Create the data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    valid_datagen = ImageDataGenerator(rescale=1./255)

    # Create the training data generator
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_roi_df,
        x_col='roi_path',
        y_col='label',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    # Create the validation data generator
    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid_roi_df,
        x_col='roi_path',
        y_col='label',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary'
    )

    # Define the model
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Optimizer
    optimizer = Adam(1e-4)

    # Compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy', AUC(), Precision(), Recall()])

    # Print model summary
    model.summary()

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-5)

    # Calculate steps per epoch
    steps_per_epoch = len(train_generator)  # Number of batches in the training data
    validation_steps = len(valid_generator)  # Number of batches in the validation data

    history = model.fit(
        train_generator,
        steps_per_epoch=int(steps_per_epoch),  # Ensure this is an integer
        epochs=30,
        validation_data=valid_generator,
        validation_steps=int(validation_steps),  # Ensure this is also an integer
        shuffle=True,
        callbacks=[early_stopping, reduce_lr]
    )

    # Evaluate the model on the test set
    test_loss, test_acc,_,i,a = model.evaluate(test_generator, verbose=1)
    print(f"Test accuracy: {test_acc}")

    model.save('GLAUCOMA_DETECTION.h5')