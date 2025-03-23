# **Capstone Project: Question Difficulty Classification**  

## **Overview**  
This project focuses on developing a machine learning model to classify the difficulty level of questions. The goal is to build a classifier that can be implemented to provide grade level and taxonomical level of a question. Current efforts involve expanding the training process to include Bloom's Taxonomy to further classify information. 
<br>
The original framework for this project is based on the research paper, *"Question Difficulty Estimation Based on Attention Model for Question Answering"* by Hyun-Je Song, Su-Hwan Yoon, and Seong-Bae Park.  
<br>
PyTorch was used wherever possible, and when functions were taken from PyTorch Docs it is notated in the code.  
<br>
<br>

## **Features**  
- **Question Preprocessing**: Tokenization, stopword removal, and feature extraction.  
- **Custom Dataset Integration**: Implementation of a dataset of Common Core Standard questions, categorized by grade level.  
- **Model Training**: Fine-tuning models for difficulty classification using both general and standardized educational content.  
- **Evaluation Metrics**: Accuracy, loss, reproducibility, and comparison across different difficulty levels.  
<br>
<br>

## **Technologies Used**  
- **Programming Languages & Libraries**: Python (Pandas, Jupyter, PyTorch, SciKit Learn)  
- **Machine Learning Models**: BERT, DUMA, Multiclass Regression, and other NLP-based classification  
- **Experimentation & Analysis**: Google Colab and Jupyter Notebook for iterative development and evaluation  
<br>
<br>

## **Data**  
The original dataset used was the RACE dataset mentioned in the paper, with difficulty levels ranging from easy (Middle School) to hard (High School).  Once able to replicate the model, additional classification questions were needed. 
- RACE Dataset --> https://huggingface.co/papers/1704.04683
<br>
<br>
The latest update includes a custom dataset of Common Core Standard questions sorted by grade level.  These questions come from a variety of textbooks that align with common core state standards.
- QxGrade Dataset --> https://www.kaggle.com/datasets/coreyjjness94/qxgrade-dataset
<br>
<br>
Using a multiclass regression logarithm, the model was trained until it tested at 75% accuracy on unseen data. 
- Studybot model --> 
<br>
<br>

## **Future Plans**  
- **Feature Engineering Improvements**: Explore additional features to enhance classification using Bloom's Taxonomy.   
- **Model Comparisons**: Evaluate different transformer-based and deep learning models for optimized performance.  
- **Application & Deployment**: Investigate potential real-world applications, such as an educational pedagogy assessment tool.  
<br>
<br>

## **Project Status**  
This project is actively evolving, with ongoing improvements in feature selection, model tuning, and dataset integration for enhanced accuracy and real-world applicability.  
<br>
<br>

## **Contact**  
For questions, collaborations, or contributions, reach out via [GitHub](https://github.com/coreyjness).  
<br>
Notice:  This project is for educational purposes only.  The data used to train this model is copyrighted and only used under fair usage rights.  


<br><br><br><br>
<br><br><br><br>

## **How To:** 