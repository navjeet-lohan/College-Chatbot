# NITJ Chatbot

## Setup
pip install torch transformers numpy nltk python-dotenv
python -m nltk.downloader punkt
pip install scikit-learn

## Training
python augment_intents.py <br>
python train_bert.py 

## Running
python chatbot.py  # Interactive
### OR
python app.py  # Web server

## Handling errors
change self.confidence threshold in chatbot.py

![image](https://github.com/user-attachments/assets/3de35874-cf96-4026-b6ec-fcbf9a56d7fa)
