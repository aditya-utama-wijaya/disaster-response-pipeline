# Disaster Response Pipeline Project

This project aims to develop a disaster response pipeline that can categorize messages received during a crisis, allowing for efficient distribution of aid and resources. The pipeline includes data preprocessing, machine learning model training, and a web application for message classification.

### Introduction

During a disaster or crisis, numerous messages are generated from various sources such as social media, emergency hotlines, and news outlets. Classifying and categorizing these messages manually can be time-consuming and inefficient. This project addresses this problem by building a machine learning pipeline to automate the process of message classification, enabling rapid response and allocation of resources.

### File Structure
The project directory contains the following files and directories:
1. `data`: Within this directory, you will find scripts for data processing as well as the raw data files:
    - `process_data.py`: This script handles the ETL (Extract, Transform, Load) process, cleaning the data and storing it in a SQLite database.
    - `disaster_messages.csv`: A CSV file that holds the disaster response messages.
    - `disaster_categories.csv`: Another CSV file that contains the categories associated with each message.
2. `models`: In this directory, you'll discover the machine learning script and the saved model file:
    - `train_classifier.py`: This script focuses on training a classifier using the cleaned data and then saves the trained model as a pickle file.
    - `classifier.pkl`: The saved model file that stores the trained classifier.
3. `app`: This directory is dedicated to the web application files:
    - `run.py`: A Python script that executes the web application.
    - `templates`: A folder containing HTML templates used for various web pages.

### Usage

To run this project, follow these steps:
1. Clone the repository

    - `git clone https://github.com/aditya-utama-wijaya/disaster-response-pipeline.git`

2. Change into the project directory

    - `cd disaster-response-pipeline`

3. To process the data and train the classifier, run the following command:

    - `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
  
    - `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

4. To launch the web application, run the following command:

    - `python app/run.py`

5. Once the server is running, open your web browser and go to http://localhost:3000 to access the web application. The web application allows you to enter a message and view the corresponding categories predicted by the trained classifier.

### Data

The project uses two datasets provided by [Figure Eight](https://www.figure-eight.com) that contain pre-labeled disaster-related messages and their respective categories:
- `disaster_messages.csv`: Contains the messages.
- `disaster_categories.csv`: Contains the message categories.

### Author
[Adi Wijaya](https://www.linkedin.com/in/aditya-utama-wijaya/)
