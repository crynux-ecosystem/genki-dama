import random
import os
import pandas as pd
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen
from pathlib import Path

from genki.validator.model_evaluator.music.music_labels import mood_theme_classes, instrument_classes


class MusicEvaluator(object):
    def __init__(
        self,
        model_id: str
    ) -> None:
        self.model = MusicGen.get_pretrained(model_id)

    @classmethod
    def generate_evaluation_prompts(cls, style: str, csv_filename: str, num: int = 10) -> str:

        print(f"style: {style}, csv_filename: {csv_filename}, num: {num}")

        with open(csv_filename, "w") as f:
            f.write("caption\n")
            for _ in range(num):
                f.write(MusicEvaluator._generate_prompt(style) + "\n")


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

    def get_music_quality_score(self, prompt_csv_filename: str, music_folder: str,  text_column: str = "caption") -> float:
        return 1.0
        
    def get_music_style_score(self, music_filename: str) -> float:
        return 1.0
    
    def compare_music_similarity(self, music_a_filename: str, music_b_filename: str) -> float:
        return 1.0


if __name__ == "__main__":
    
    eval_folder = Path(__file__).parent.parent.parent.parent.parent / "evaluation"

    prompts_file = eval_folder / "prompts.csv"
    music_folder = eval_folder / "musics"
    

    print("generating prompts...")

    MusicEvaluator.generate_evaluation_prompts(
        "Electronic---Chiptune",
        prompts_file
    )

    df = pd.read_csv(prompts_file)
    text_data = df["caption"].tolist()

    evaluator = MusicEvaluator(model_id="iwehf/my_musicgen")

    print("generating musics...")

    i = 0
    for prompt in text_data:
        
        print(f"{i+1} / {len(text_data)}")

        evaluator.text_2_music(
            prompt,
            os.path.join(music_folder, f"music_{i}")
        )

        i+=1

    print("music quality score: ")
    print(evaluator.get_music_quality_score(prompts_file, music_folder))
