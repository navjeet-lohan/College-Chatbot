
## Setup
pip install torch transformers numpy nltk python-dotenv
python -m nltk.downloader punkt
pip install scikit-learn

## Training
python train_bert.py 
python augment_intents.py 

## Running
python chatbot.py  # Interactive
# OR
python app.py  # Web server
