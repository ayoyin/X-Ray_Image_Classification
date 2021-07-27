<div align="center">
  <h1>Chest X-Ray Images (Pneumonia)</h1>
  
  <h3>Created by Ayotunde Yoyin</h3>
</div>


## Introduction

As we adapt and evolve as a species, automation is a large part of this change that we experience that allows us to grow technically, and also opens up a new depth to our problem solving capabilites. Automation has the ability to help us maintain discipline, make decisions, and provide advice, along with many other benefits. This project attempts to take automation to the next level within the medical field in providing diagnoses to patients experiencing symptons of pneumonia. The aim of this project is to create a binary classification model that inputs an image of a chest x-ray and outputs a diagnosis of whether or not the inidividual has pneumonia. 

## Background & Problem Statement

In detecting pneumonia in patients, the typical procedure is to listen to the patient's lungs using a stethoscope to observe any abnormalities, hypothesizing a visual diagnosis of a patient's chest x-ray, and if necessary for a clearer diagnosis, using a blood test to confirm the presence of the infection.

**PROBLEM STATEMENT:** Create a model that can accurately detect the presence of a pnemonia infection given an image of a patient's chest x-ray.

## Data

[Click here for Chest X-Ray Images (Pneumonia) Kaggle Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

The data used for this project was sourced from a Kaggle.com competition. It includes 5,216 images for training, 624 images for testing, and 16 images for validation.

Among the 5216 images for TRAINING, there were 1,341 images of NORMAL lungs while there were 3,875 images of lungs with PNEUMONIA.

Among the 624 images for TESTING, there were 233 images of NORMAL lungs while there were 389 images of lungs with PNEUMONIA.

The validation set was split 50-50 between NORMAL and PNEUMONIA. 8 NORMAL images and 8 PNEUMONIA images.

![12ex](https://user-images.githubusercontent.com/44102000/127099166-8e842d4b-da3b-488e-ba6d-3dc48c6c6a9c.png)

## Modeling & Analysis
The approach in modeling was to use a Convolutional Neural Network (CNN). This is because CNNs are adept at image recognition because of their application of the convolution and pooling layers which work together to turn an image into an array of information represent the patterns and colors that occur within a given image.

![CNN](https://user-images.githubusercontent.com/44102000/127189698-6b597885-05ef-4ad7-87b6-a73ae6fdc183.jpeg)

The aim in modeling was to construct a model which will iterate over the NORMAL and PNEUMONIA training images to learn the differences between an instance of NORMAL lungs vs. and instance of PNEUMONIA lungs.

### RESNET50V2
Pre-trained CNN with 50 layers having a deep architecture for image classification.
```
model = Sequential([
    ResNet50V2(include_top = False, pooling='avg', weights='imagenet'),
    layers.BatchNormalization(),
#     layers.Dense(256, activation='relu'),
#     layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid'),
])

model.layers[0].trainable = False

model.summary()
adam = optimizers.Adam(learning_rate = 0.001, beta_1 = 0.22, beta_2 = 0.999)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
```

```
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode = 'min', restore_best_weights = True, patience=3)
m1 = model.fit(train_set, epochs = 100, validation_data = test_set, callbacks = [callback], validation_steps = 12, steps_per_epoch = 7)

m1

Epoch 1/100
7/7 [==============================] - 9s 1s/step - loss: 0.5817 - accuracy: 0.7455 - val_loss: 0.6030 - val_accuracy: 0.6276
Epoch 2/100
7/7 [==============================] - 6s 1s/step - loss: 0.3342 - accuracy: 0.8705 - val_loss: 0.4044 - val_accuracy: 0.8724
Epoch 3/100
7/7 [==============================] - 7s 1s/step - loss: 0.2034 - accuracy: 0.9286 - val_loss: 0.4300 - val_accuracy: 0.8203
Epoch 4/100
7/7 [==============================] - 7s 1s/step - loss: 0.2644 - accuracy: 0.9062 - val_loss: 0.3712 - val_accuracy: 0.8203
Epoch 5/100
7/7 [==============================] - 7s 1s/step - loss: 0.1874 - accuracy: 0.9330 - val_loss: 0.3812 - val_accuracy: 0.8776
Epoch 6/100
7/7 [==============================] - 7s 1s/step - loss: 0.1852 - accuracy: 0.9509 - val_loss: 0.3541 - val_accuracy: 0.8568
Epoch 7/100
7/7 [==============================] - 7s 1s/step - loss: 0.1428 - accuracy: 0.9777 - val_loss: 0.3674 - val_accuracy: 0.8255
Epoch 8/100
7/7 [==============================] - 6s 1s/step - loss: 0.1730 - accuracy: 0.9598 - val_loss: 0.4981 - val_accuracy: 0.7526
Epoch 9/100
7/7 [==============================] - 7s 1s/step - loss: 0.1150 - accuracy: 0.9732 - val_loss: 0.4353 - val_accuracy: 0.7969
```

### Classification Report & Confusion Matrix
```
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(test_df['class'], y_pred, target_names=labels))
print(confusion_matrix(test_df['class'], y_pred))

df_matrix=pd.DataFrame(confusion_matrix(test_df['class'], y_pred), 
             columns=["Predicted Normal", "Predicted Pneumonia"], 
             index=["Actual Normal", "Actual Pneumonia"])

df_matrix

               precision    recall  f1-score   support

      NORMAL       0.92      0.80      0.86       234
   PNEUMONIA       0.89      0.96      0.92       390

    accuracy                           0.90       624
   macro avg       0.91      0.88      0.89       624
weighted avg       0.90      0.90      0.90       624
```

![confusion](https://user-images.githubusercontent.com/44102000/127200849-652ed7cf-1a90-4491-885b-1b9a44ba49d0.png)


```markdown
Syntax highlighted code block


## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```



For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/ayoyin/X-Ray_Image_Classification/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
