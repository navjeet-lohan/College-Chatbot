import json
import random
import torch  # Added this import
from transformers import pipeline
from tqdm import tqdm

class IntentAugmenter:
    def __init__(self):
        # Initialize with device detection
        self.device = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline(
            'text-generation', 
            model='gpt2-medium',
            device=self.device
        )

    def augment_intent(self, intent, samples=15):
        base_prompts = random.sample(intent['patterns'], min(5, len(intent['patterns'])))
        prompt = (
            f"Generate {samples} natural variations of these questions about {intent['tag']} "
            f"for a technical university chatbot:\n" + 
            "\n".join(f"- {p}" for p in base_prompts) +
            "\n-"
        )
        
        results = self.generator(
            prompt,
            max_length=60,
            num_return_sequences=1,
            temperature=0.75,
            top_k=50
        )
        
        return [
            line.strip()[2:] 
            for line in results[0]['generated_text'].split('\n') 
            if line.startswith('- ') and 10 < len(line) < 100
        ][:samples]

def main():
    augmenter = IntentAugmenter()
    
    with open('intents.json') as f:
        data = json.load(f)

    for intent in tqdm(data['intents'], desc="Augmenting Intents"):
        if len(intent['patterns']) < 20:
            try:
                new_patterns = augmenter.augment_intent(intent)
                intent['patterns'].extend(new_patterns)
            except Exception as e:
                print(f"Error augmenting {intent['tag']}: {str(e)}")
                continue

    with open('intents_augmented.json', 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()