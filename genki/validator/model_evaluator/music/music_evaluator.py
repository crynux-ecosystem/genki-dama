import random
import os
from typing import List
import pandas as pd
import requests
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen
from pathlib import Path

from genki.validator.model_evaluator.music.music_labels import mood_theme_classes, instrument_classes

score_endpoint = "http://127.0.0.1:5000"

class MusicEvaluator(object):
    def __init__(
        self,
        model_id: str
    ) -> None:
        self.model = MusicGen.get_pretrained(model_id)

    @classmethod
    def generate_evaluation_prompts(cls, style: str, csv_filename: str, num: int = 10) -> List[str]:

        print(f"style: {style}, csv_filename: {csv_filename}, num: {num}")

        prompts = []

        with open(csv_filename, "w") as f:
            f.write("caption\n")
            for _ in range(num):
                prompt = MusicEvaluator._generate_prompt(style)
                prompts.append(prompt)
                f.write(f"{prompt}\n")

        return prompts   

    @classmethod
    def _generate_prompt(cls, style: str) -> str:
        mood = random.sample(mood_theme_classes, 3)
        instruments = random.sample(instrument_classes, 2)
        return style + " " + " ".join(mood) + " ".join(instruments)

    def text_2_music(
        self,
        text_prompt: str,
        filename: str,
        use_sampling: bool = True,
        top_k: int = 250,
        top_p: float = 0.0,
        temperature: float = 1.0,
        duration: float = 30.0,
        cfg_coef: float = 3.0,
        two_step_cfg: bool = False,
        extend_stride: float = 18,
    ):
        self.model.set_generation_params(
            use_sampling=use_sampling,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            duration=duration,
            cfg_coef=cfg_coef,
            two_step_cfg=two_step_cfg,
            extend_stride=extend_stride,
        )

        wavs = self.model.generate(descriptions=[text_prompt])

        audio_write(
            filename,
            wavs[0],
            self.model.sample_rate,
            format="wav",
            strategy="loudness",
            loudness_compressor=True,
        )

    def get_music_quality_score(self, prompt_csv_filename: str, music_folder: str,  text_column: str = "caption") -> tuple[float, float]:
        query_params = {
            'csv_file': prompt_csv_filename,
            'music_folder': music_folder,
            'text_column': text_column
        }

        response = requests.get(score_endpoint + "/clap", params=query_params)

        if response.status_code != 200:
            print(f"request failed to get clap scores: {response.status_code}")
            raise Exception(f"request failed to get clap scores: {response.status_code}")

        response_json = response.json()

        return response_json["mean"], response_json["std"]


    def get_music_style_score(self, music_filename: str) -> float:
        return 1.0


    @classmethod
    def compare_music_similarity(cls, music_a_filename: str, music_b_filename: str) -> float:
        query_params = {
            'music_folder_a': music_a_filename,
            'music_folder_b': music_b_filename
        }

        response = requests.get(score_endpoint + "/similarity", params=query_params)

        if response.status_code != 200:
            print(f"request failed to get similarity scores: {response.status_code}")
            raise Exception(f"request failed to get similarity scores: {response.status_code}")

        response_json = response.json()

        return response_json


if __name__ == "__main__":

    eval_folder = Path(__file__).parent.parent.parent.parent.parent / "evaluation"

    prompts_file = eval_folder / "prompts.csv"
    music_folder = eval_folder / "musics"

    ft_music_folder = music_folder / "ft"
    ft2_music_folder = music_folder / "ft2"
    original_music_folder = music_folder / "original"

    # print("generating prompts...")

    # MusicEvaluator.generate_evaluation_prompts(
    #     "Electronic---Chiptune",
    #     prompts_file
    # )

    # df = pd.read_csv(prompts_file)
    # text_data = df["caption"].tolist()

    evaluator_ft = MusicEvaluator(model_id="iwehf/my_musicgen")
    evaluator_ft2 = MusicEvaluator(model_id="iwehf/pokemon_musicgen")
    evaluator_original = MusicEvaluator(model_id="facebook/musicgen-small")

    # print("generating musics...")

    # i = 0
    # for prompt in text_data:

    #     evaluator_ft.text_2_music(
    #         prompt,
    #         os.path.join(ft_music_folder, f"music_{i}")
    #     )

    #     evaluator_ft2.text_2_music(
    #         prompt,
    #         os.path.join(ft2_music_folder, f"music_{i}")
    #     )

    #     evaluator_original.text_2_music(
    #         prompt,
    #         os.path.join(original_music_folder, f"music_{i}")
    #     )

    #     print(f"{i+1} / {len(text_data)}")
    #     i+=1

    print("ft model quality score: ")
    print(evaluator_ft.get_music_quality_score(prompts_file, ft_music_folder))

    print("ft2 model quality score: ")
    print(evaluator_ft2.get_music_quality_score(prompts_file, ft2_music_folder))

    print("original model quality score: ")
    print(evaluator_original.get_music_quality_score(prompts_file, original_music_folder))

    print("original VS ft similarity: ")
    print(MusicEvaluator.compare_music_similarity(original_music_folder, ft_music_folder))

    print("original VS ft2 similarity: ")
    print(MusicEvaluator.compare_music_similarity(original_music_folder, ft2_music_folder))

    print("ft VS ft2 similarity: ")
    print(MusicEvaluator.compare_music_similarity(ft_music_folder, ft2_music_folder))
