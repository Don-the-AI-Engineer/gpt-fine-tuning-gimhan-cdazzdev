import os
from openai import OpenAI
import time
from dotenv import load_dotenv
from config import BASE_MODEL

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def upload_training_file(file_path='training_examples.jsonl'):
    """Upload the training file to OpenAI"""
    with open(file_path, "rb") as f:
        uploaded_file = client.files.create(
            file=f,
            purpose="fine-tune"
        )
    return uploaded_file.id

def create_fine_tuning_job(file_id, model=BASE_MODEL):
    """Create and start a fine-tuning job"""
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=model
    )
    return job.id

def monitor_job_status(job_id):
    """Monitor the status of the fine-tuning job"""
    while True:
        job_info = client.fine_tuning.jobs.retrieve(job_id)
        status = job_info.status
        print("Fine-tuning job status:", status)
        
        # Get recent events
        events = client.fine_tuning.jobs.list_events(
            fine_tuning_job_id=job_id,
            limit=10
        )
        for event in events.data:
            print(event.message)
            
        if status in ["succeeded", "failed", "cancelled"]:
            break
        time.sleep(10)  # wait 10 seconds before checking again

    if status == "succeeded":
        model_name = job_info.fine_tuned_model
        print("✅ Fine-tuned model name:", model_name)
        return model_name
    else:
        print("❌ Fine-tuning did not succeed. Status:", status)
        return None

def train_model():
    """Main training function"""
    # Upload training file
    print("Uploading training file...")
    file_id = upload_training_file()
    print("✅ File uploaded successfully:", file_id)

    # Create fine-tuning job
    print("Creating fine-tuning job...")
    job_id = create_fine_tuning_job(file_id)
    print("🎯 Fine-tuning job started:", job_id)

    # Monitor progress and get model name
    print("Monitoring training progress...")
    model_name = monitor_job_status(job_id)
    
    return model_name

if __name__ == "__main__":
    model_name = train_model()
    if model_name:
        print("Training completed successfully. Model name:", model_name)
        # Save model name to file for testing
        with open("model_name.txt", "w") as f:
            f.write(model_name)