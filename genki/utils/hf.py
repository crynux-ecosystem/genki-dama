from genki.model.creative_model import CreativeModel
from typing import Tuple

def validate_hf_repo_id(repo_id: str) -> Tuple[str, str]:
    """Verifies a Hugging Face repo id is valid and returns it split into namespace and name.

    Raises:
        ValueError: If the repo id is invalid.
    """

    if not repo_id:
        raise ValueError("Hugging Face repo id cannot be empty.")

    if not 3 < len(repo_id) <= CreativeModel.MAX_REPO_ID_LENGTH:
        raise ValueError(f"Hugging Face repo id must be between 3 and {CreativeModel.MAX_REPO_ID_LENGTH} characters.")

    parts = repo_id.split("/")
    if len(parts) != 2:
        raise ValueError("Hugging Face repo id must be in the format <org or user name>/<repo_name>.")

    return parts[0], parts[1]
