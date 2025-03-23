# **Capstone Project: Question Difficulty Classification**  

## **Overview**  
This project focuses on developing a machine learning model to classify the difficulty level of questions. The goal is to build a classifier that predicts both the grade level and taxonomical complexity of a question. Current efforts involve expanding the training process to include Bloom's Taxonomy to further classify information. 
<br>
The original framework for this project is based on the research paper, *"Question Difficulty Estimation Based on Attention Model for Question Answering"* by Hyun-Je Song, Su-Hwan Yoon, and Seong-Bae Park.  
<br>
PyTorch was used extensively, with explicit annotations in the code when functions were sourced from official documentation.

<br>

## **Features**  
- **Question Preprocessing**: Tokenization, stopword removal, and feature extraction.  
- **Custom Dataset Integration**: Implementation of a dataset of Common Core Standard questions, categorized by grade level.  
- **Model Training**: Fine-tuning models for difficulty classification using both general and standardized educational content.  
- **Evaluation Metrics**: Accuracy, loss, reproducibility, and comparison across different difficulty levels.  

<br>

## **Technologies Used**  
- **Programming Languages & Libraries**: Python (Pandas, Jupyter, PyTorch, SciKit Learn)  
- **Machine Learning Models**: BERT, DUMA, Multiclass Regression, and other NLP-based classification  
- **Experimentation & Analysis**: Google Colab and Jupyter Notebook for iterative development and evaluation  

<br>

## **Data**  
The original dataset used was the RACE dataset mentioned in the paper, with difficulty levels ranging from easy (Middle School) to hard (High School).  Once able to replicate the model, additional classification questions were needed. 

- RACE Dataset --> https://huggingface.co/papers/1704.04683

<br>
The latest update includes a custom dataset of Common Core Standard questions sorted by grade level.  These questions come from a variety of textbooks that align with common core state standards.

- QxGrade Dataset --> https://www.kaggle.com/datasets/coreyjjness94/qxgrade-dataset

<br>
A multiclass classification model was trained to achieve approximately 75% accuracy on unseen data.

- Studybot model --> https://huggingface.co/Coreyjness/Studybot
<br>


## **Future Plans**  
- **Feature Engineering Improvements**: Explore additional features to enhance classification using Bloom's Taxonomy.   
- **Model Comparisons**: Evaluate different transformer-based and deep learning models for optimized performance.  
- **Application & Deployment**: Investigate potential real-world applications, such as an educational pedagogy assessment tool.  
<br>
<br>

## **Project Status**  
This project is actively maintained, with continuous enhancements in feature engineering, model optimization, and dataset curation to improve accuracy and applicability.  
<br>
Previous model iterations can be found in the mtversions folder, and the process I took to create the dataset is in the qxgrade_dataset folder.  
<br>
<br>

## **Contact**  
For questions, collaborations, or contributions, reach out via [GitHub](https://github.com/coreyjness).  
<br>
Notice:  This project is for educational purposes only.  The data used to train this model is copyrighted and only used under fair usage rights.  


<br><br><br><br>
<br><br><br><br>

## **How To:** 

-  1. Download the dataset as QxGrade_Dataset.csv if you are training.  Download the model as Bert_Classifier.pt if you are deploying the model.
<br>

-  2. Create a fork in the repository and download the project to a studybot folder so you can access the notebook locally.  
<br>

- 3. Spin up the virtual environment specific to your system. 
<br>

- 4.  Download the required libraries using:  pip install -r requirements.txt