import os
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def load_system_message():
    """Load the system message from the training data"""
    with open('training_examples.jsonl', 'r') as f:
        first_line = f.readline()
        import json
        first_example = json.loads(first_line)
        return first_example['messages'][0]['content']

def test_model(model_name=None, num_tests=5):
    """Test the fine-tuned model with sample prompts"""
    if model_name is None:
        try:
            with open("model_name.txt", "r") as f:
                model_name = f.read().strip()
        except FileNotFoundError:
            print("❌ No model name provided and couldn't find model_name.txt")
            return

    df = pd.read_json('training_examples.jsonl', lines=True)
    system_message = load_system_message()

    print(f"Testing model: {model_name}")
    print(f"System message: {system_message}")
    print("\nRunning test cases:")
    print("-" * 50)

    test_prompts = df['messages'].apply(lambda x: x[1]['content']).sample(n=num_tests)
    
    for i, test_prompt in enumerate(test_prompts, 1):
        print(f"\nTest case {i}:")
        print(f"Prompt: {test_prompt}")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": test_prompt,
                }
            ]
        )
        
        print(f"Response:\n{response.choices[0].message.content}")
        print("-" * 50)

if __name__ == "__main__":
    test_model()