import argparse
import json
import os
import re

import librosa
import numpy as np
from essentia.standard import (MonoLoader, TensorflowPredict2D,
                               TensorflowPredictEffnetDiscogs)
from pydub import AudioSegment

from genki.music_labels import (genre_labels, instrument_classes,
                                mood_theme_classes)


def resample_and_filter(dataset_path: str, result_path: str, duration: int, ext: str):
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)

    filenames = os.listdir(dataset_path)
    for filename in filenames:
        if filename.endswith(ext):
            filepath = os.path.join(dataset_path, filename)
            audio: AudioSegment = AudioSegment.from_file(filepath)

            audio = audio.set_frame_rate(44100)
            if len(audio) >= duration * 1000:
                audio.export(
                    os.path.join(result_path, filename[:-4] + ".wav"), format="wav"
                )


def filter_predictions(predictions, class_list, threshold=0.1):
    predictions_mean = np.mean(predictions, axis=0)
    sorted_indices = np.argsort(predictions_mean)[::-1]
    filtered_indices = [i for i in sorted_indices if predictions_mean[i] > threshold]
    filtered_labels = [class_list[i] for i in filtered_indices]
    filtered_values = [predictions_mean[i] for i in filtered_indices]
    return filtered_labels, filtered_values


def make_comma_separated_unique(tags):
    seen_tags = set()
    result = []
    for tag in tags:
        if tag not in seen_tags:
            result.append(tag)
            seen_tags.add(tag)
    return result


def get_audio_features(audio_filename: str):
    result_dict = {}

    audio = MonoLoader(filename=audio_filename, sampleRate=16000, resampleQuality=4)()
    embedding_model = TensorflowPredictEffnetDiscogs(
        graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1"
    )
    embeddings = embedding_model(audio)

    genre_model = TensorflowPredict2D(
        graphFilename="genre_discogs400-discogs-effnet-1.pb",
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0",
    )
    predictions = genre_model(embeddings)
    filtered_labels, _ = filter_predictions(predictions, genre_labels)
    filtered_labels = ", ".join(filtered_labels).replace("---", ", ").split(", ")
    result_dict["genres"] = make_comma_separated_unique(filtered_labels)

    mood_model = TensorflowPredict2D(
        graphFilename="mtg_jamendo_moodtheme-discogs-effnet-1.pb"
    )
    predictions = mood_model(embeddings)
    filtered_labels, _ = filter_predictions(
        predictions, mood_theme_classes, threshold=0.05
    )
    result_dict["moods"] = make_comma_separated_unique(filtered_labels)

    # predict instruments
    instrument_model = TensorflowPredict2D(
        graphFilename="mtg_jamendo_instrument-discogs-effnet-1.pb"
    )
    predictions = instrument_model(embeddings)
    filtered_labels, _ = filter_predictions(predictions, instrument_classes)
    result_dict["instruments"] = filtered_labels

    return result_dict


def tag_audio(dataset_path: str):
    audio_filenames = [
        filename for filename in os.listdir(dataset_path) if filename.endswith(".wav")
    ]
    for filename in audio_filenames:
        title, _ = os.path.splitext(filename)
        filepath = os.path.join(dataset_path, filename)
        audio_features = get_audio_features(filepath)
        # get key and BPM
        y, sr = librosa.load(filepath)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = round(tempo.tolist()[0])  # not usually accurate lol
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key = np.argmax(np.sum(chroma, axis=1))
        key = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][key]
        length = librosa.get_duration(y=y, sr=sr)
        # populate json

        title_content = re.sub(r"\d+|\.", "", title).strip()
        genre = audio_features.get("genres", [])
        moods = audio_features.get("moods", [])
        instruments = audio_features.get("instruments", [])
        description = "A chiptune music"
        if len(moods) > 0:
            moods_str = ", ".join(moods)
            description += f", with {moods_str} theme"

        entry = {
            "key": f"{key}",
            "sample_rate": sr,
            "file_extension": "wav",
            "description": description,
            "keywords": "",
            "duration": length,
            "bpm": tempo,
            "genre": genre,
            "title": title_content,
            "name": "",
            "instrument": instruments,
            "moods": moods,
        }
        tag_file = os.path.join(dataset_path, title + ".json")
        with open(tag_file, mode="w", encoding="utf-8") as f:
            json.dump(entry, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="source audio files directory")
    parser.add_argument(
        "--dst", required=True, help="preprocessed dataset files directory"
    )
    parser.add_argument(
        "--duration", default=30, help="minimum audio length threshold in seconds"
    )
    parser.add_argument("--ext", default="wav", help="source audio files ext")

    args = parser.parse_args()

    resample_and_filter(args.src, args.dst, args.duration, args.ext)
    tag_audio(args.dst)
