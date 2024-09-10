from flask import current_app

def get_similarity_score(folder1, folder2):
    return current_app.config["frechet"].score(folder1, folder2)
