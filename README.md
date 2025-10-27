# GPT-3.5 Fine-Tuning

This is the Job Assessment for the CDAZZDEV Senior Machine Learning Engineer Position.

This project implements a fine-tuned GPT-3.5 model that translates natural language commands into ROS2 and Unitree SDK Python code. The system takes high-level commands (e.g., "Go to the charging dock and report battery level") and generates appropriate Python code sequences.

## Assignment Methodology

The assignment was to follow the fine-tuning pipeline demonstrated in the `mshumer/gpt-llm-trainer` repository.

This approach involves a "one-prompt" methodology:

1. **Define a Task**: A single, high-level prompt is created to define the model's new, specialized skill.
2. **Generate Data**: This prompt is fed to a powerful "teacher" model (GPT-4) to synthetically generate a high-quality, task-specific dataset (e.g., 100+ prompt/response pairs).
3. **Fine-Tune**: This new synthetic dataset is used to fine-tune a more cost-effective "student" model (GPT-3.5 Turbo), making it an expert at the defined task.

This project is a clean-room implementation of that pipeline, refactored into a structured Python application.

## Model Selection Rationale

My initial approach was to use the Llama-2 model, as it is open-source and well-suited for deployment on edge devices like the robotics hardware mentioned in my CV. However, I encountered significant library incompatibility issues between the required versions for Llama-2 (e.g., bitsandbytes, peft, trl) and the modern Google Colab environment.

These version conflicts made the Llama-2 path prohibitively time-consuming for this assessment.

Therefore, to demonstrate a successful implementation of the full pipeline (data generation, fine-tuning, and inference), I pivoted to using GPT-3.5 Turbo. This model is supported via a stable API, allowing me to focus on implementing the core logic of the assignment.

## Prerequisites

- Python 3.9  
- Conda package manager  
- OpenAI API key  

## Environment Setup

Create a new conda environment:

```bash
conda create -n robotics-gpt python=3.9
conda activate robotics-gpt
```

Install required packages:

```bash
pip install openai pandas python-dotenv tenacity
```

Set up your environment variables:

1. Create a `.env` file in the project root  
2. Add your OpenAI API key:

```text
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

- `config.py`: Configuration file containing model parameters and settings  
- `data_generation.py`: Generates training data for fine-tuning  
- `training.py`: Handles the model fine-tuning process  
- `testing.py`: Tests the fine-tuned model  
- `training_examples.jsonl`: Generated training data (will be created during execution)  

## Configuration

The `config.py` file contains all configurable parameters:

- `MODEL_PROMPT`: Defines the behavior of the model  
- `TEMPERATURE`: Controls response randomness (0.0-1.0)  
- `NUMBER_OF_EXAMPLES`: Number of training examples to generate  
- `BASE_MODEL`: The model to be fine-tuned (`gpt-3.5-turbo`)  
- `DATA_MODEL`: Model used for generating training data (`gpt-4`)  

## Use Case Rationale & Prompt

To align this project with my professional experience in robotics (including work with the Unitree GO2, ROS2, and LLM-based function calling), I chose a highly specialized task: translating natural language into executable robotics code.

The `MODEL_PROMPT` in `config.py` is the core of this assignment:

> "A model that acts as a specialized Robotics SDK Translator. It will be given a high-level, natural language command from a user (e.g., 'Go to the charging dock and then report your battery level'). It must respond with a synthetically correct, logical sequence of Python code, specifically using ROS2 functions and Unitree SDK commands for navigation and system-checking. The response must be only the code, formatted as a Python list of command strings."

## Usage

### Generate Training Data:

```bash
python data_generation.py
```

This script:

- Feeds the `MODEL_PROMPT` to the `DATA_MODEL` (GPT-4) to generate diverse training examples.  
- Creates a system message for the model.  
- Saves the formatted data in JSONL format.  

### Train the Model:

```bash
python training.py
```

This script:

- Uploads the training data to OpenAI.  
- Initiates the fine-tuning process on the `BASE_MODEL`.  
- Monitors training progress.  
- Saves the fine-tuned model name.  

### Test the Model:

```bash
python testing.py
```

This script:

- Loads the fine-tuned model name.  
- Runs test cases with sample prompts.  
- Displays the model's expert responses.  

## Expected Output (Proof of Success)

When running `python testing.py`, the script will feed a novel, unseen prompt to the fine-tuned model. The expected output demonstrates the model has specialized in its task:

**Test Prompt (Input):**

"I need the robot to go to the kitchen, find the red ball, and then come back to the living room."

**Fine-Tuned Model Output (Result):**

```python
[
  "self.nav_publisher.publish(self.create_nav_goal(room='kitchen'))",
  "self.wait_for_navigation()",
  "self.perception_client.call_async(self.create_perception_request(task='find_object', object_name='red ball'))",
  "self.wait_for_perception()",
  "self.nav_publisher.publish(self.create_nav_goal(room='living_room'))",
  "self.wait_for_navigation()"
]
```

This shows the model correctly translated the complex, multi-step command into the specific Python code it was trained to produce, rather than a general conversational answer.

## Important Notes

### API Key Security:

- Never commit your `.env` file to version control.  
- Add `.env` to your `.gitignore` file.  
- Users must provide their own OpenAI API key.  

### Cost Awareness:

- Data generation (`data_generation.py`) uses GPT-4 (more expensive).  
- Fine-tuning and inference (`training.py`, `testing.py`) use GPT-3.5-turbo (more cost-effective).  
- Monitor your API usage to control costs.  

### Customization:

- Modify `MODEL_PROMPT` in `config.py` to change the model's behavior.  
- Adjust `TEMPERATURE` and `NUMBER_OF_EXAMPLES` based on your needs.  
- Fine-tune hyperparameters for optimal results.  

## Error Handling

- The data generation script includes retry logic for API failures.  
- Training progress is monitored with status updates.  
- Testing includes proper error handling for missing model files.

### Please note that ChatBots were used to furnish this Readme file.