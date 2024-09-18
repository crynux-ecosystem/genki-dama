import argparse
import os

from huggingface_hub import HfApi


def upload(access_token: str, repo_id: str, src: str):
    api = HfApi(token=access_token)

    api.upload_folder(repo_id=repo_id, folder_path=src, repo_type="model")


if __name__ == "__main__":
    access_token = os.getenv("HF_ACCESS_TOKEN")
    assert access_token is not None

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="huggingface repo id to upload the model",
    )
    parser.add_argument(
        "--src", type=str, required=True, help="local model source directory to upload"
    )
    args = parser.parse_args()

    upload(access_token, repo_id=args.repo_id, src=args.src)
