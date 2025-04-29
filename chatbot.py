import torch
import json
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import warnings
warnings.filterwarnings("ignore")

class CollegeChatbot:
    def __init__(self, model_path='./model'):
        try:
            print("Initializing chatbot...")  # Debug line
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")  # Debug line
            
            # Load model
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
            print("Model loaded successfully")  # Debug line
            
            # Load intents
            with open('intents_augmented.json', 'r', encoding='utf-8') as f:
                self.intents = json.load(f)
            print(f"Loaded {len(self.intents['intents'])} intents")  # Debug line
            
            # Load tag mappings
            with open(f'{model_path}/tag2id.json', 'r') as f:
                self.tag2id = json.load(f)
            self.id2tag = {int(v): k for k, v in self.tag2id.items()}
            print("Tag mappings loaded")  # Debug line
            
            self.confidence_threshold = 0.20
            print("Chatbot initialized successfully\n")  # Debug line
            
        except Exception as e:
            print(f"\nCRITICAL INIT ERROR: {str(e)}\n")
            raise

    def predict_intent(self, text):
        try:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            probs = torch.softmax(outputs.logits, dim=1)[0]
            confidence, pred_id = torch.max(probs, dim=0)
            return self.id2tag[pred_id.item()], confidence.item()
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return "error", 0.0

    def get_response(self, intent_tag):
        try:
            for intent in self.intents['intents']:
                if intent['tag'] == intent_tag:
                    return intent['responses']
            return [{"type": "text", "text": "I don't have information about that."}]
        except Exception as e:
            print(f"Response error: {str(e)}")
            return [{"type": "text", "text": "Error processing request."}]

    def process_query(self, text):
        intent, confidence = self.predict_intent(text)
        print(f"\nDebug: '{text}' -> {intent} (confidence: {confidence:.2f})")  # Debug line
        
        if confidence < self.confidence_threshold:
            return [{'type': 'text', 'text': "Please visit main NITJ website for more detailed information"}]
        
        responses = self.get_response(intent)
        return responses if responses else [{'type': 'text', 'text': "No response available."}]

def main():
    print("\n" + "="*50)
    print("NITJ Chatbot System".center(50))
    print("="*50 + "\n")
    
    try:
        bot = CollegeChatbot()
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                responses = bot.process_query(user_input)
                
                print("\nBot:")  # Ensure output is visible
                for response in responses:
                    if isinstance(response, str):
                        print(response)
                    elif isinstance(response, dict):
                        print(response.get('text', ''))
                        if 'url' in response:
                            print(f"More Details: {response['url']}")
                print()  # Add spacing
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
                
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        print("Please check:")
        print("- Model files exist in ./model")
        print("- intents_augmented.json is valid")
        print("- Required packages are installed\n")

if __name__ == "__main__":
    main()