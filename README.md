# NITJ Chatbot

<img src="https://github.com/user-attachments/assets/89b64d00-3a5a-4197-a580-963d51c20f0d" width="50%" />
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


