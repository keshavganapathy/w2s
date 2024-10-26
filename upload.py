import sys
from huggingface_hub import HfApi, upload_folder

def main():
    if len(sys.argv) != 2:
        print("Usage: python upload_model.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]

    # Read the Hugging Face token from the 'hf_token' file
    try:
        with open('hf_token', 'r') as f:
            token = f.read().strip()
    except FileNotFoundError:
        print("Error: 'hf_token' file not found.")
        sys.exit(1)

    # Initialize the Hugging Face API
    api = HfApi()

    # Retrieve the username associated with the token
    try:
        user_info = api.whoami(token=token)
        username = user_info['name']
    except Exception as e:
        print(f"Error retrieving user info: {e}")
        sys.exit(1)

    repo_id = f"{username}/{model_name}"

    # Create the repository on Hugging Face Hub
    try:
        api.create_repo(name=model_name, token=token, exist_ok=True)
        print(f"Repository '{repo_id}' is ready.")
    except Exception as e:
        print(f"Error creating repository '{repo_id}': {e}")
        sys.exit(1)

    # Upload the 'output' folder to the repository
    try:
        upload_folder(
            folder_path='output',
            path_in_repo='.',  # Upload to the root of the repository
            repo_id=repo_id,
            token=token,
            commit_message="Upload model",
            ignore_patterns=["*.ipynb_checkpoints", "__pycache__"],
        )
        print(f"Model uploaded successfully to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Error uploading the model to '{repo_id}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()