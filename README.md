# GPT-3.5 Fine-Tuning for Robotics SDK Translation

This is the Job Assessment for CDAZZDEV Senior Machine Learning Engineer Position

This project implements a fine-tuned GPT-3.5 model that translates natural language commands into ROS2 and Unitree SDK Python code. The system takes high-level commands (e.g., "Go to the charging dock and report battery level") and generates appropriate Python code sequences.

## Prerequisites

- Python 3.9
- Conda package manager
- OpenAI API key

## Environment Setup

1. Create a new conda environment:
```bash
conda create -n robotics-gpt python=3.9
conda activate robotics-gpt
```

2. Install required packages:
```bash
pip install openai pandas python-dotenv tenacity
```

3. Set up your environment variables:
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
   ```
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
- `BASE_MODEL`: The model to be fine-tuned (gpt-3.5-turbo)
- `DATA_MODEL`: Model used for generating training data (gpt-4)

## Usage

1. Generate Training Data:
```bash
python data_generation.py
```
This script:
- Generates diverse training examples using GPT-4
- Creates a system message for the model
- Saves the formatted data in JSONL format

2. Train the Model:
```bash
python training.py
```
This script:
- Uploads the training data to OpenAI
- Initiates the fine-tuning process
- Monitors training progress
- Saves the fine-tuned model name

3. Test the Model:
```bash
python testing.py
```
This script:
- Loads the fine-tuned model
- Runs test cases with sample prompts
- Displays the model's responses

## Important Notes

1. **API Key Security**: 
   - Never commit your `.env` file to version control
   - Add `.env` to your `.gitignore` file
   - Users must provide their own OpenAI API key

2. **Cost Awareness**:
   - Data generation uses GPT-4 (more expensive)
   - Fine-tuning and inference use GPT-3.5-turbo (more cost-effective)
   - Monitor your API usage to control costs

3. **Customization**:
   - Modify `MODEL_PROMPT` in `config.py` to change the model's behavior
   - Adjust `TEMPERATURE` and `NUMBER_OF_EXAMPLES` based on your needs
   - Fine-tune hyperparameters for optimal results

## Error Handling

- The data generation includes retry logic for API failures
- Training progress is monitored with status updates
- Testing includes proper error handling for missing model files