## 🧠 Student Performance Prediction Project 📊

This project, "Student Exam Performance Indicator," showcases a comprehensive **Machine Learning (ML) pipeline** designed to predict student performance based on various academic and demographic attributes. It emphasizes the journey from raw data to a deployed, interactive web application, making predictions accessible to users.

---

### 🌟 Project Overview

The core objective of this project is to predict a student's **Maths Score** out of 100, leveraging their demographic information (Gender, Race or Ethnicity), parental education, lunch type, test preparation course completion, and scores in Reading and Writing. The project demonstrates a full-stack ML solution, from data processing to real-time prediction via a user-friendly web interface.

<img width="783" height="897" alt="image" src="https://github.com/user-attachments/assets/9cadc6e1-7fa4-409e-8b18-f002a67a9216" />



---

### ✨ Key Features

This project incorporates the following crucial stages of a machine learning workflow:

* **Data Ingestion**: Automates the fetching and loading of raw data into the system. This ensures that the model always trains on relevant and up-to-date information.
* **Data Transformation**: Implements essential preprocessing steps, including **scaling** numerical features and effectively handling **categorical features** to prepare the data for model training.
* **Model Training**: Utilizes various machine learning algorithms to train and rigorously **evaluate performance models**. This stage focuses on selecting the most accurate and robust model for predictions.
* **Prediction Pipeline**: Provides a **seamless integration** for real-time predictions based on new input data. This allows the web application to query the trained model efficiently.
* **Web Deployment**: Features a **user-friendly interface** powered by **Flask**, enabling easy interaction with the prediction model. Users can input data and receive instant score predictions.

![Student Performance Prediction Project Details](<img width="783" height="897" alt="image" src="https://github.com/user-attachments/assets/6b920eab-1d4e-44cf-a1aa-6f45812d69c6" />
)

---

### 💻 Technologies Used

* **Python**: The primary programming language for the entire ML pipeline and web application.
* **Flask**: A micro web framework used for developing the lightweight and interactive web interface.
* **Scikit-learn**: A robust machine learning library for model training, evaluation, and data preprocessing (e.g., scaling, encoding).
* **Pandas**: Essential for data manipulation and analysis.
* **NumPy**: Used for numerical operations, especially with arrays.

---

### 🚀 Getting Started

Follow these steps to set up and run the Student Performance Prediction Project locally:

#### Prerequisites

* Python 3.x installed on your system.

#### Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/shubhamparulekar/mlproject.git](https://github.com/shubhamparulekar/mlproject.git)
    cd mlproject
    ```
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you have a `requirements.txt` file in your repository listing all necessary libraries.)*

#### Running the Application

1.  **Launch the Flask web application**:
    ```bash
    python app.py
    ```
    *(Note: The main application file might be named differently, e.g., `main.py` or `run.py`.)*
2.  Open your web browser and navigate to the address displayed in your terminal (e.g., `http://127.0.0.1:5000/`).

---

### 💡 Usage

Once the application is running, you can interact with it through the web interface:

1.  **Input Data**: On the "Student Exam Performance Indicator" page, select or enter values for the following features:
    * Gender
    * Race or Ethnicity
    * Parental Level of Education
    * Lunch Type
    * Test Preparation Course
    * Reading Score out of 100
    * Writing Score out of 100
2.  **Predict**: Click the **"Predict your Maths Score"** button.
3.  **View Prediction**: The predicted Maths Score will be displayed on the page under "The prediction is".

<img width="783" height="897" alt="image" src="https://github.com/user-attachments/assets/0f93e982-1528-4ee9-bf97-dcd202d1b9de" />


---

### 🌐 Deployed Application

You can also access a live version of the Student Performance Prediction Project deployed on Render:

**[https://studentperformanceprediction-8inu.onrender.com/](https://studentperformanceprediction-8inu.onrender.com/)**

---

### 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add new feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request.

---

### 📄 Acknowledgements

This readme.md was generated using Gemini

---
