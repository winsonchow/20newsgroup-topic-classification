# 20 Newsgroups Topic Classification

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Installation and Setup](#installation-and-setup)
4. [Project Structure](#project-structure)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Development](#model-development)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Results](#results)
9. [Challenges and Improvements](#challenges-and-improvements)
10. [Future Work](#future-work)
11. [Acknowledgements](#acknowledgements)

## Project Overview
This project involves building and evaluating a machine learning model using the BERT (Bidirectional Encoder Representations from Transformers) architecture to classify text documents into one of twenty different newsgroups.

## Dataset Description
The dataset used for this project is the 20 Newsgroups dataset. It consists of approximately 20,000 newsgroup documents, partitioned across 20 different newsgroups.

## Installation and Setup
To set up the project environment, follow these steps:
1. Clone the repository: `git clone https://github.com/winsonchow/20newsgroups-topic-classification.git`
2. Navigate to the project directory: `cd 20newsgroups-topic-classification`
3. Install the required libraries: `pip install -r requirements.txt`

## Project Structure
- `20newsgroups-topic-classification.ipynb`: The main notebook containing the entire project code.
- `requirements.txt`: The list of libraries required to run the project.

## Data Preprocessing
The text data is preprocessed to fit the input requirements of the BERT model. This involves:
- Tokenizing the text data.
- Padding and truncating the sequences to ensure uniform input size.
- Creating attention masks to distinguish between actual tokens and padding tokens.

## Model Development
A pre-trained BERT model is fine-tuned on the 20 Newsgroups dataset to understand the context and content of the discussion threads. The model is trained to classify each document into one of the 20 newsgroups.

## Evaluation Metrics
The model's performance is evaluated using a hold-out test set that the model has not seen during training. The primary metrics for evaluation include:
- Accuracy
- Precision
- Recall
- F1 Score

## Results
The BERT model demonstrates high accuracy in classifying text documents into the correct newsgroups, showcasing the effectiveness of contextual word representations.
- Accuracy: 0.8621
- Precision: 0.8649
- Recall: 0.8621
- F1 Score: 0.8621

## Challenges and Improvements
Possible improvements include:
- Experimenting with different preprocessing techniques.
- Fine-tuning BERT model hyperparameters for better performance.

## Future Work
Future work could involve:
- Exploring other transformer-based models.
- Implementing techniques to handle imbalanced classes.
- Extending the model to classify documents from other datasets or domains.

## Acknowledgements
Special thanks to Professor Lim for his guidance and support.