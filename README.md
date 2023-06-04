# Disaster Response Pipeline Project

This project aims to develop a disaster response pipeline that can categorize messages received during a crisis, allowing for efficient distribution of aid and resources. The pipeline includes data preprocessing, machine learning model training, and a web application for message classification.

### Introduction

During a disaster or crisis, numerous messages are generated from various sources such as social media, emergency hotlines, and news outlets. Classifying and categorizing these messages manually can be time-consuming and inefficient. This project addresses this problem by building a machine learning pipeline to automate the process of message classification, enabling rapid response and allocation of resources.

### Usage

To run this project, follow these steps:
1. Clone the repository

  `git clone https://github.com/aditya-utama-wijaya/disaster-response-pipeline.git`

2. Change into the project directory

  `cd disaster-response-pipeline`

3. To process the data and train the classifier, run the following command:

  `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`

4. To launch the web application, run the following command:

  `python run.py`

5. Once the server is running, open your web browser and go to http://localhost:3000 to access the web application. The web application allows you to enter a message and view the corresponding categories predicted by the trained classifier.

### Data

The project uses two datasets provided by [Figure Eight](https://www.figure-eight.com) that contain pre-labeled disaster-related messages and their respective categories:
- `disaster_messages.csv`: Contains the messages.
- `disaster_categories.csv`: Contains the message categories.

### Author
[Adi Wijaya](https://www.linkedin.com/in/aditya-utama-wijaya/)
