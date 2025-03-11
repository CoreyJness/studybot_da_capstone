# **Capstone Project: Question Difficulty Classification**  

## **Overview**  

This project focuses on developing a machine learning model to classify the difficulty level of questions. The goal is to build a classifier that can be implemented to provide grade level and taxonomical level of a question. Current efforts involve expanding the training process to incorporate a custom dataset consisting of Common Core Standard questions sorted by grade level.  

The original framework for this project is based on the research paper, *"Question Difficulty Estimation Based on Attention Model for Question Answering"* by Hyun-Je Song, Su-Hwan Yoon, and Seong-Bae Park.  

## **Features**  

- **Question Preprocessing**: Tokenization, stopword removal, and feature extraction.  
- **Custom Dataset Integration**: Implementation of a dataset of Common Core Standard questions, categorized by grade level.  
- **Model Training**: Fine-tuning models for difficulty classification using both general and standardized educational content.  
- **Evaluation Metrics**: Accuracy, loss, reproducibility, and comparison across different difficulty levels.  

## **Technologies Used**  

- **Programming Languages & Libraries**: Python (Pandas, Jupyter, PyTorch, NumPy)  
- **Machine Learning Models**: BERT, DUMA, and other NLP-based classification  
- **Experimentation & Analysis**: Jupyter Notebook for iterative development and evaluation  

## **Data**  
The original dataset was the RACE dataset mentioned in the paper, with difficulty levels ranging from easy(Middle School) to hard(High School).  Once able to replicate the model (99% in testing versus the 97% stated in the paper), additional classification questions were necessary.
The latest update includes a custom dataset of Common Core Standard questions sorted by grade level.


## **Future Plans**  

- **Fine-tuning with Custom Data**: Continue training the model using the Common Core dataset to improve domain-specific accuracy.  
- **Feature Engineering Improvements**: Explore additional linguistic and cognitive features to enhance classification precision.  
- **Model Comparisons**: Evaluate different transformer-based and deep learning models for optimized performance.  
- **Application & Deployment**: Investigate potential real-world applications, such as an adaptive learning assistant or an educational assessment tool.  

## **Project Status**  

This project is actively evolving, with ongoing improvements in feature selection, model tuning, and dataset integration for enhanced accuracy and real-world applicability.  In depth walk-though instructions and explanations will be provided when the project is closer to finishing. 

## **Contact**  

For questions, collaborations, or contributions, reach out via [GitHub](https://github.com/coreyjness).  
