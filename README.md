# Sentiment Analysis Of FIFA World Cup Tweets

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mfehr7/tweet-sentiment-analysis/blob/main/sentiment-analysis.ipynb)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project is a Recurrent Neural Network (RNN) designed to analyze 2022 FIFA World Cup Tweets and predict their sentiment. Specifically, this implements the Long Short-Term Memory architecture. I chose this project to deepen my understanding of Natural Language Processing, as well as the usage of RNNs on sequential data. This framework can also be expanded to time-series data among other things. The specific dataset was chosen because I have a passion for soccer and watching the World Cup! 

## Data
I used a dataset with ~22,000 rows, taken from Hugging Face. Each observation is a tweet and it's corresponding sentiment (positive, neutral, or negative).

**Dataset:** Tirendaz, A. (2022). FIFA World Cup 2022 Tweets. Hugging Face. Retrieved from: https://huggingface.co/datasets/Tirendaz/fifa-world-cup-2022-tweets

## Deep Learning Process

To analyze this data and train the neural network, I followed these steps:

1.  **EDA**: I started by learning more about the data, including looking at the data structure, checking for duplicates, and visualizing the distribution of the sentiment labels (positive, neutral, negative). This step is crucial as I found that there were duplicates which could've caused overfitting in the network, and the label counts were imbalanced.
2. **Preprocessing**: To prepare my data for training, I cleaned the tweets by removing web addresses, new line characters, user mentions, and more. Then, I tokenized the tweets and padded them to equal lengths. Finally, the sentiment labels were converted to integer values.
3. **Model Training/Architecture**: I used a Long Short-Term Memory (LSTM) network architecture with the following layers:
    * **Embedding (10000, 32):** Takes the tokenized text and transforms each number into a vector.
    * **Bidirectional LSTM (32):** Remembers context from earlier in a sequence to make better predictions. Returns sequences so that next LSTM layer can use them.
    * **Dropout (0.4):** Generalization technique that randomly turns off 40% of the neurons in the previous layer.
    * **Bidirectional LSTM (32):** Remembers context from earlier in a sequence to make better predictions. Doesn't return sequences, just outputs a vector.
    * **Dropout (0.4):** Generalization technique that randomly turns off 40% of the neurons in the previous layer.
    * **Dense, ReLU activation (16):** Standard, fully connected layer.
    * **Dropout (0.4):** Generalization technique that randomly turns off 40% of the neurons in the previous layer.
    * **Dense, Softmax activation (3):** Standard, fully connected layer accompanied by softmax to convert raw output into a probability distribution. When using the network, the label corresponding to the highest probability will be predicted.  
    
    A learning rate of 0.0001 was chosen, due to super quick convergence (leading to overfitting) with a higher learning rate. Early stopping was also added to train the model, based on the validation accuracy.
4. **Final Evaluation**: The model was evaluated on a separate test set to determine how well it generalizes to new, unseen data.

## Technologies Used

* **Language:** Python
* **Core Framework:** TensorFlow
* **Environment:** Google Colab

## Getting Started

You can run this project in two ways:

### Option 1: Run in Google Colab

This is the easiest way to get started.

1.  **Click the Badge:** Click the "Open in Colab" badge at the top of this README.
2.  **Run the Notebook:** Execute the cells in order from top to bottom.

### Option 2: Run Locally

If you prefer to run the project on your own machine, follow these steps.

1.  **Prerequisites:**
    * Python 3.10 or higher
    * Git
    * Conda
    * JupyterLab

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/mfehr7/sentiment-analysis.git
    cd tweet-sentiment-analysis
    ```

3.  **Activate your Conda environment:**
    * **Windows:** Open Anaconda Prompt and run `conda activate [environment-name]`
    * **macOS/Linux:** Open Terminal and run `conda activate [environment-name]`
    * Make sure that JupyterLab is installed in your active environment

4.  **Launch JupyterLab**
    * In your Anaconda Prompt (or Terminal), run the following command:

    ```bash
    jupyter lab
    ```

    This will start the JupyterLab server and should automatically open a new tab in your web browser. If it doesn't, your terminal will provide a URL (usually `http://localhost:8888/...`) that you can copy and paste into your browser.

5.  **Run the Notebook**

    * From the file browser in JupyterLab, click on your notebook file (`sentiment-analysis.ipynb`).
    * Once the notebook is open, run the cells in order from top to bottom.
