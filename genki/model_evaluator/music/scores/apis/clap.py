from flask import current_app

def get_clap_score(csv_file, music_folder, text_column):
    return current_app.config["clap"].score(
        text_path=csv_file,
        audio_dir=music_folder,
        text_column=text_column,
        text_embds_path=None,
        audio_embds_path=None,
    )
