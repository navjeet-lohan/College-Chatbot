# NITJ Chatbot

## Setup
pip install torch transformers numpy nltk python-dotenv
python -m nltk.downloader punkt
pip install scikit-learn

## Training
python augment_intents.py 
python train_bert.py 

## Running
python chatbot.py  # Interactive
### OR
python app.py  # Web server

## Handling errors
change self.confidence threshold in chatbot.py