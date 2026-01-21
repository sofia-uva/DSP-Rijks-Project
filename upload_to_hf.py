from huggingface_hub import HfApi
import os

api = HfApi()

repo_id = "sofiaspero/rijks_features"

api.upload_file(
    path_or_fileobj="DATA/final_multimodal_features.npy",
    path_in_repo="final_multimodal_features.npy",
    repo_id=repo_id,
    repo_type="model",
    token=os.environ["HF_TOKEN"],
)

api.upload_file(
    path_or_fileobj="DATA/final_features.npy",
    path_in_repo="final_features.npy",
    repo_id=repo_id,
    repo_type="model",
    token=os.environ["HF_TOKEN"],
)

print("Upload complete!")