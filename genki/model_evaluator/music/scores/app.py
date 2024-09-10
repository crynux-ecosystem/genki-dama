from flask import Flask, request, jsonify
from apis.clap import get_clap_score
from apis.similarity import get_similarity_score
from frechet_audio_distance import CLAPScore, FrechetAudioDistance

from pathlib import Path


app = Flask(__name__)


def init_models():
    model_cache_dir = Path(__file__).parent / "models"

    app.config['clap'] = CLAPScore(
        ckpt_dir=model_cache_dir
    )

    app.config['frechet'] = FrechetAudioDistance(
        ckpt_dir=model_cache_dir
    )


@app.route('/clap', methods=['GET'])
def clap():
    csv_file = request.args.get('csv_file', '')
    music_folder = request.args.get('music_folder', '')
    text_column = request.args.get('text_column', 'caption')
    score_mean, score_std = get_clap_score(csv_file, music_folder, text_column)
    return jsonify({"mean": float(score_mean), "std": float(score_std)})


@app.route('/similarity', methods=['GET'])
def similarity():
    folder1 = request.args.get('music_folder_a', '')
    folder2 = request.args.get('music_folder_b', '')
    result = get_similarity_score(folder1, folder2)
    return jsonify({"similarity": float(result)})


if __name__ == "__main__":
    init_models()
    app.run(debug=True)
