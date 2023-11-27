from cog import BasePredictor, Input
import transformers


MODEL_PATH = 'models'
MODEL_NAME = 'spicyboros-c34b-2.2'


class Predictor(BasePredictor):
    tokenizer: transformers.AutoTokenizer
    model: transformers.AutoModelForCausalLM

    def setup(self) -> None:
        path = f"{MODEL_PATH}/{MODEL_NAME}"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(path)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(path)

    def predict(
        self,
        prompt: str = Input(description="Model Prompt")
    ) -> str:
        generator = transformers.pipeline(
            task="text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer,
        )

        result = generator(
            prompt,
            max_length=512, 
            num_return_sequences=1,
        )

        return result[0]['generated_text']
