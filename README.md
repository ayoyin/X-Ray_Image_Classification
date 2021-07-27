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
The approach in modeling was to use a Convolutional Neural Network (CNN). This is because CNNs are adept 
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
