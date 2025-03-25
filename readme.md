
# **Capstone Project: Question Difficulty Classification**

## **Overview**
This project focuses on building a machine learning model that classifies the difficulty of a given question by predicting its **grade level (3rd–12th)**. Future enhancements aim to include **Bloom’s Taxonomy levels**, enabling the model to estimate cognitive complexity as well.

The architecture is inspired by *"Question Difficulty Estimation Based on Attention Model for Question Answering"* by Hyun-Je Song, Su-Hwan Yoon, and Seong-Bae Park.

The project is implemented in **PyTorch**, with heavy customization to the BERT architecture through a **Dual Attention Mechanism (DualBert)**. Code includes inline references to PyTorch documentation where appropriate.

---

## **Key Features**
- 🔁 **Custom BERT Architecture**: Implements a dual self-attention mechanism (`DualBertModel`) to better capture question semantics.
- 🧠 **Grade-Level Classification**: Predicts question difficulty as one of ten grade levels (3rd to 12th).
- 🧮 **Custom Dataset Integration**: Uses a dataset of Common Core-aligned questions labeled by grade.
- 📊 **Evaluation**: Tracks accuracy and loss across multiple training sessions.
- 🚀 **Streamlit App**: Deploy the trained model via an interactive UI to classify user-input questions in real-time.

---

## **Technologies Used**
- **Languages & Tools**: Python, PyTorch, Hugging Face Transformers, scikit-learn, Jupyter Notebook, Streamlit  
- **Models & Architectures**: BERT, Dual Attention (custom), Multiclass Classification  
- **Development Environment**: Jupyter Notebook, Google Colab (for training), Streamlit (for deployment)

---

## **Data**
- 📚 **RACE Dataset** (original benchmark from the paper)  
  - [Hugging Face RACE Paper](https://huggingface.co/papers/1704.04683)

- 📘 **QxGrade Dataset** (custom dataset curated from Common Core-aligned educational material)  
  - [Kaggle: QxGrade Dataset](https://www.kaggle.com/datasets/coreyjjness94/qxgrade-dataset)

- 🤖 **Pretrained Model**  
  - [Hugging Face: Studybot](https://huggingface.co/Coreyjness/Studybot)  
    Achieves ~70% accuracy on unseen grade-level questions.

---

## **How to Use the Project**

Follow the instructions below to train your own model or run the pre-trained classifier using Streamlit.

---

### 🧰 Setup Instructions

1. **Download the Resources**  
   - If training: download `QxGrade_Dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/coreyjjness94/qxgrade-dataset)  
   - If deploying: download `Bert_Classifier.pt` from [Hugging Face](https://huggingface.co/Coreyjness/Studybot)
<br>
2. **Clone the Repository and Set Up Your Environment**
   ```bash
   git clone https://github.com/coreyjness/studybot_da_capstone.git
   cd studybot_da_capstone
   ```
<br>
3. **(Optional) Create and Activate a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
<br>
4. **Install Required Libraries**

    You can install the dependencies based on your system:
    
    🔹 CPU-only  
    If you're running on a regular laptop or desktop without a dedicated NVIDIA GPU (note: training will not run on most CPUs:
    
    ```bash
    pip install -r requirements.txt
    ```
    
    🔹 GPU Acceleration (NVIDIA with CUDA 11.8)
    
    ```bash
    pip install -r requirements_gpu.txt
    ```
<br>
5. **Launch Jupyter Notebook from the command line to Train a New Model**
   ```bash
   jupyter notebook
   ```
<br>
6. **Train the Classifier**
   - Open `capstonev2.ipynb`
   - Run through the notebook to load and preprocess the dataset, fine-tune the `DualBertModel`, and save the resulting model as `Bert_Classifier.pt`
<br>
7. **Launch the Classifier with Streamlit**
   ```bash
   streamlit run main.py
   ```

---

## **Project Status**
This project is actively maintained, with ongoing enhancements to:

- Model performance  
- Dataset quality  
- Bloom’s Taxonomy integration  
- Interface usability  

Previous versions are archived in the `mtversions/` folder. The dataset building process is documented in `qxgrade_dataset/`.

---

## **Contact**
For questions, collaborations, or contributions, reach out via [GitHub](https://github.com/coreyjness).

---

## ⚠️ **License & Usage**
This project is for educational and research purposes only.  
All training data is either publicly available or used under fair use doctrine. Please do not redistribute or commercialize this project without appropriate rights or citation.
