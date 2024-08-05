# Breast Cancer Cell Classification using Support Vector Machine (SVM)
# Project Overview
This project involves developing a Support Vector Machine (SVM) classifier to predict whether breast cancer cell samples are benign or malignant. The dataset contains various attributes of cell samples, and the classifier uses these attributes to determine the nature of the samples. This project showcases my skills in machine learning and data analysis, forming part of my portfolio.

# Dataset
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Data Set. It includes the following columns:
ID: Identifier for the sample
Clump: Clump Thickness
UnifSize: Uniformity of Cell Size
UnifShape: Uniformity of Cell Shape
MargAdh: Marginal Adhesion
SingEpiSize: Single Epithelial Cell Size
BareNuc: Bare Nuclei
BlandChrom: Bland Chromatin
NormNucl: Normal Nucleoli
Mit: Mitoses
Class: Classification (2 for benign, 4 for malignant)
The dataset is sourced from an IBM course data repository.

# Project Steps

1. Data Loading and Exploration
The project begins with loading the dataset and performing exploratory data analysis (EDA) to understand its structure and contents. This involves checking the first few rows of the dataset and summarizing the data to identify any initial patterns or issues.

2. Data Cleaning
A crucial step in preparing the dataset is handling any non-numeric values, particularly in the BareNuc column, which contains some non-numeric entries. These values are converted to numeric, and any rows with missing or invalid data are removed to ensure the dataset is clean and ready for analysis.

3. Feature Selection
After cleaning the data, relevant features are selected for model training. In this case, attributes such as Clump, UnifSize, UnifShape, MargAdh, SingEpiSize, BareNuc, BlandChrom, NormNucl, and Mit are chosen as the features. The class label Class is used as the target variable.

4. Data Splitting
The dataset is split into training and testing sets to evaluate the model's performance. Typically, 80% of the data is used for training, and 20% is reserved for testing. This ensures that the model can be trained on a substantial amount of data while being tested on unseen data to assess its accuracy.

5. Model Training and Prediction
The SVM algorithm is employed to train the model. Two types of SVM kernels are used: radial basis function (RBF) and linear. The model is trained on the training dataset, and predictions are made on the testing dataset using both kernels. This allows for comparison between the two approaches.

6. Model Evaluation
The model's performance is evaluated using various metrics, including precision, recall, F1-score, and the Jaccard index. These metrics provide a comprehensive view of the classifier's effectiveness. Additionally, a confusion matrix is used to visualize the model's performance, showing the true versus predicted classifications.

7. Confusion Matrix Visualization
To further understand the model's performance, a confusion matrix is plotted. This matrix displays the number of true positive, true negative, false positive, and false negative predictions, providing insight into the model's accuracy and potential areas for improvement.

# Results

The performance of the SVM classifier with both RBF and linear kernels is evaluated. The results indicate high accuracy in classifying the breast cancer cells. Specifically:

RBF Kernel:
F1-Score: 0.96
Jaccard Index: 0.94
Linear Kernel:
F1-Score: 0.96
Jaccard Index: 0.94
Both kernels show similar performance, demonstrating the robustness of the SVM approach in this classification task.

# Conclusion

The SVM classifier, using both RBF and linear kernels, demonstrates high accuracy in predicting whether a breast cancer cell sample is benign or malignant. This project highlights the application of machine learning techniques in the medical field, specifically for cancer diagnosis. The successful implementation and high performance of the model underscore its potential utility in real-world medical diagnostics.

# Requirements

To run this project, the following tools and libraries are required:

Python 3.x
pandas
numpy
matplotlib
scikit-learn
