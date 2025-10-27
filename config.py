# Model configuration
MODEL_PROMPT = """A model that acts as a specialized Robotics SDK Translator. 
                It will be given a high-level, natural language command from a user 
                (e.g., 'Go to the charging dock and then report your battery level'). 
                It must respond with a syntactically correct, logical sequence of 
                Python code, specifically using ROS2 functions and Unitree SDK commands 
                for navigation and system-checking. The response must be only the code, 
                formatted as a Python list of command strings."""

# Training parameters
TEMPERATURE = 0.2
NUMBER_OF_EXAMPLES = 100
N_RETRIES = 3

# Model settings
BASE_MODEL = "gpt-3.5-turbo"
DATA_MODEL = "gpt-4"  # Model used for generating training data