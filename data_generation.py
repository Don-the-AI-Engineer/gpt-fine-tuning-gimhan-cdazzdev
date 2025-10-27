import os
import random
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
import json
import pandas as pd
from dotenv import load_dotenv
from config import MODEL_PROMPT, TEMPERATURE, NUMBER_OF_EXAMPLES, N_RETRIES, DATA_MODEL

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

@retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=70))
def generate_example(prompt, prev_examples, temperature=TEMPERATURE):
    messages=[
        {
            "role": "system",
            "content": f"You are generating data which will be used to train a machine learning model.\n\nYou will be given a high-level description of the model we want to train, and from that, you will generate data samples, each with a prompt/response pair.\n\nYou will do so in this format:\n```\nprompt\n-----------\n$prompt_goes_here\n-----------\n\nresponse\n-----------\n$response_goes_here\n-----------\n```\n\nOnly one prompt/response pair should be generated per turn.\n\nFor each turn, make the example slightly more complex than the last, while ensuring diversity.\n\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model.\n\nHere is the type of model we want to train:\n`{prompt}`"
        }
    ]

    if len(prev_examples) > 0:
        if len(prev_examples) > 8:
            prev_examples = random.sample(prev_examples, 8)
        for example in prev_examples:
            messages.append({
                "role": "assistant",
                "content": example
            })

    response = client.chat.completions.create(
        model=DATA_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=1000,
    )

    return response.choices[0].message.content

def generate_system_message(prompt, temperature=TEMPERATURE):
    response = client.chat.completions.create(
        model=DATA_MODEL,
        messages=[
          {
            "role": "system",
            "content": "You will be given a high-level description of the model we are training, and from that, you will generate a simple system prompt for that model to use. Remember, you are not generating the system message for data generation -- you are generating the system message to use for inference. A good format to follow is `Given $INPUT_DATA, you will $WHAT_THE_MODEL_SHOULD_DO.`.\n\nMake it as concise as possible. Include nothing but the system prompt in your response.\n\nFor example, never write: `\"$SYSTEM_PROMPT_HERE\"`.\n\nIt should be like: `$SYSTEM_PROMPT_HERE`."
          },
          {
              "role": "user",
              "content": prompt.strip(),
          }
        ],
        temperature=temperature,
        max_tokens=500,
    )

    return response.choices[0].message.content

def create_training_data():
    # Generate examples
    prev_examples = []
    for i in range(NUMBER_OF_EXAMPLES):
        print(f'Generating example {i}')
        example = generate_example(MODEL_PROMPT, prev_examples, TEMPERATURE)
        prev_examples.append(example)

    # Generate system message
    system_message = generate_system_message(MODEL_PROMPT)
    print(f'System message: {system_message}')

    # Process examples into DataFrame
    prompts = []
    responses = []

    for example in prev_examples:
        try:
            split_example = example.split('-----------')
            prompts.append(split_example[1].strip())
            responses.append(split_example[3].strip())
        except:
            pass

    df = pd.DataFrame({
        'prompt': prompts,
        'response': responses
    })

    df = df.drop_duplicates()
    print(f'Generated {len(df)} unique examples')

    # Create training examples in JSONL format
    training_examples = []
    for index, row in df.iterrows():
        training_example = {
            "messages": [
                {"role": "system", "content": system_message.strip()},
                {"role": "user", "content": row['prompt']},
                {"role": "assistant", "content": row['response']}
            ]
        }
        training_examples.append(training_example)

    # Save to file
    with open('training_examples.jsonl', 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')

    return system_message

if __name__ == "__main__":
    system_message = create_training_data()
    print("✅ Training data generated and saved to training_examples.jsonl")