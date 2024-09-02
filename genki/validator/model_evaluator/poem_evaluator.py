
from genki.validator.model_evaluator.api.api import GPTAPI
from genki.validator.model_evaluator.api.claude_api import ClaudeAPI

GET_EVALUATION_THEME_PROMPT = '''
Please generate a random theme for writing a poem.
The theme should inspire creativity and evoke emotion, suitable for exploring through poetry.
Just return the theme, do not return anything else
'''

GET_POEM_PROMPT='''
Please write a poem using the following theme:
'''

GET_RESULT_PROMPT = '''
Please evaluate the following poem based on the following theme. Consider the following aspects:
1. Depth and authenticity of emotional expression: Does the poem resonate emotionally, capturing complex and nuanced human feelings?
2. Exploration and expression of the theme: How does the author navigate and articulate the theme, offering unique insights or profound exploration?
3. Creativity and originality: Does the poem demonstrate innovation in concept, word choice, or format?
4. Aesthetic appeal and technical skill: Is the language of the poem beautiful, employing rhetorical devices such as rhyme, metaphor, symbolism, etc., to enhance its expressive power?
5. Structure and fluency: Is the poem's structure clear, supporting the theme's presentation, and does it read smoothly and naturally?
Based on the above criteria, please assign a comprehensive score between 1 and 100 to this poem, where 100 represents perfection and 1 represents very poor. Return only an int type number as the evaluation score.
'''

class PoemEvaluator:

    def __init__(self, api: GPTAPI):
        self.api = api

    def generate_evaluation_theme(self) -> str:
        return self.api.get_response(GET_EVALUATION_THEME_PROMPT)

    def generate_poem_writing_prompt_for_theme(self, theme: str) -> str:
        return GET_POEM_PROMPT + theme

    def evaluate_poem(self, theme: str, poem: str) -> int:

        prompt = GET_RESULT_PROMPT + f"\nThe theme is '{theme}', and the poem is '{poem}'"
        score_txt = self.api.get_response(prompt)

        try:
            score = int(score_txt)
        except:
            score = 0

        return score

if __name__ == "__main__":
    evaluator = PoemEvaluator(ClaudeAPI())
    theme = evaluator.generate_evaluation_theme()
    print(f"theme: {theme}")
    