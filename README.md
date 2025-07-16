
# AI-Driven Crop Disease Prediction and Management System 🌾

This repository hosts the code and resources for an AI-Driven Crop Disease Prediction and Management System. Designed to support farmers and agricultural experts, this solution leverages computer vision to detect crop diseases in real-time, helping users make informed decisions to enhance crop health and productivity.


## Project Overview
This system uses Convolutional Neural Networks (CNNs) to analyze crop images and environmental data, accurately identifying crop diseases. The application offers personalized treatment recommendations and actionable insights to help manage disease spread and protect crop yields. It is deployed as a web application to make it accessible to farmers in various regions.
## Features

- **Disease Detection:** Detects and identifies specific crop diseases from images.
- **Treatment Recommendations:** Provides customized recommendations based on disease type.
- **Data Analytics:** Analyzes environmental factors and integrates them into disease prediction.
- **Platform Deployment:** Available on web platform for widespread accessibility.
## Tech Stack
**Programming Language :** Python

**Libraries and Frameworks :** TensorFlow, Keras, Flask/Django (for web deployment)

**Model Architecture :** Convolutional Neural Network (CNN)
## Run Locally

To run this project locally, follow these steps:
1. Clone the Repository

```bash
  git clone https://github.com/sudhans18/AI-Driven-Crop-Disease-Management-System
  cd crop-disease-prediction
```
2. Install Dependencies
```bash
 pip install -r requirements.txt
```
3. Run the Application
Start the web application:
```bash
python app.py
```
Open your browser and go to http://localhost:5000 to access the application. 


    
## Data Set
The dataset used for this project includes images of various crops with labeled diseases. It is available publicly in [kaggle](https://kaggle.com/datasets/arjuntejaswi/plant-village). Make sure to download and place it in the data/ directory.
## Project Structure

```plaintext
├── data/                # Contains the dataset and related files
├── models/              # Pre-trained models and checkpoints
├── app.py               # Main application file
├── templates/           # HTML templates for web app
│   ├── result.html     
│   ├── upload.html  
├── style.css            # css file
├── requirements.txt     # Dependencies for the project
└── README.md            # Project documentation
```
##  Usage

- step 1: **Upload an Image-** Upload an image of the affected crop.
- step 2: **Disease Prediction-** The model will classify the disease present.
- step 3: **View Recommendations-** The system suggests possible treatments based on the disease.
## Future Improvements
- **Additional Crops and Diseases:** Expand to support more crops and diseases.
- **Real-Time Updates:** Integrate live weather data for more accurate disease predictions.
- **Advanced Analytics:** Provide insights on disease trends over time.
## Contact
For questions or collaboration inquiries, feel free to reach out:
- [LinkedIn](https://www.linkedin.com/in/yourusername)
- [Email](mailto:sudhan4843@gmail.com)
