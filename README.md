# Sign Language Image Classification using Deep Learning

This project applies **deep learning techniques** to classify sign language hand gestures into their respective categories. It was developed as part of *Using Machine Learning Tools (UMLT)*.  

The notebook demonstrates the full workflow: preparing the dataset, applying augmentation, building a convolutional neural network (CNN), training, validating, and evaluating the model.

---

## Project Overview
- **Goal**: Recognise sign language gestures from images using a convolutional neural network.  
- **Dataset**: Directory-structured image dataset of hand signs (A–Z) organised into train, validation, and test sets  
- **Key Tasks**:
  1. Load and preprocess the dataset  
  2. Apply data augmentation  
  3. Build and train a CNN model  
  4. Validate and evaluate model performance  
  5. Generate confusion matrices and prediction outputs  

---

## Settings
- **Batch size**: 8  
- **Epochs**: 20  
---

## Dataset

The dataset used in this project is too large to include in this repository.  
You can download it from [[Google Drive](https://drive.google.com/drive/u/0/folders/1orxBzKdOuXC-V1LW38QFy3iyuzoesleh)] or [[Kaggle Dataset]](https://www.kaggle.com/datasets/datamunge/sign-language-mnist).

Once downloaded, extract it and place it under the `data/` directory in the following format:

 ```bash
  data/
  train/
    A/
    B/
    ...
  val/
    A/
    B/
    ...
  test/
    A/
    B/
    ...
```
---

## Tools & Libraries Used
- **Python ≥ 3.5**  
- [tensorflow](https://www.tensorflow.org/) – deep learning framework  
- [keras](https://keras.io/) – high-level neural networks API  
- [numpy](https://numpy.org/) – numerical operations  
- [pandas](https://pandas.pydata.org/) – data handling  
- [matplotlib](https://matplotlib.org/) – visualisation  
- [scikit-learn](https://scikit-learn.org/) – evaluation utilities  
- [seaborn](https://seaborn.pydata.org/) – statistical visualisation  

---

## Notebook Structure
1. **Dataset preparation**  
2. **Data augmentation**  
3. **CNN model development**  
4. **Training and validation**  
5. **Performance evaluation**  
6. **Prediction visualisation**  

---

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/tauhid6069/sign-language-image-classification.git
   cd sign-language-image-classification
   ```
2. Install the dependencies:
```bash
pip install -r requirements.txt
```
3. Open the notebook: sign-language-image-classification.ipnyb
4. Run all cells to train and evaluate the model.
