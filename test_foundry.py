import os
import sys

# Limit CPU threads to prevent overheating/shutdown
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["ONNXRUNTIME_NUM_THREADS"] = "2"

from foundry_local_sdk import FoundryLocalManager
from openai import OpenAI

def test_inference():
    from foundry_local_sdk import Configuration
    config = Configuration(app_name="local-rag-assistant")
    manager = FoundryLocalManager(config)

    model_name = "qwen3-0.6b"
    print(f"Retrieving model '{model_name}' from catalog...")
    model = manager.catalog.get_model(model_name)
    
    print(f"Ensuring model is downloaded...")
    model.download()
    
    print(f"Loading model '{model_name}' into runtime...")
    model.load()
    
    print("Starting Foundry Local web service...")
    manager.start_web_service()
    
    base_url = manager.urls[0] + "/v1"
    loaded_model_id = "qwen3-0.6b-generic-cpu"
    
    print(f"Connecting to local endpoint at {base_url}...")
    client = OpenAI(
        base_url=base_url,
        api_key="none"
    )
    
    prompt = "Explain in one sentence why local AI models are useful."
    print(f"Sending prompt to model: '{prompt}'")
    
    try:
        response = client.chat.completions.create(
            model=loaded_model_id,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        print("\n--- Model Response ---")
        print(response.choices[0].message.content)
        print("----------------------\n")
        print("Success! Connection and inference with local model work.")
    except Exception as e:
        print(f"Error during inference: {e}")
        
if __name__ == "__main__":
    test_inference()
