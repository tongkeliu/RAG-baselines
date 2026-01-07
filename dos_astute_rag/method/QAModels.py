import os
import logging
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)


class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context, question):
        pass


class QAModel(BaseQAModel):
    def __init__(self, model_name):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.modelString = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map=device
        )

    def answer_question(self, context, meta_data):
        prompt = f"""<|im_start|>user
        Based on the following past reviews from user AE22QOZDLAODPKJHYZI2HXQ7MLZQ and the book information, write a new Amazon book review in their exact voice and style. The review must be 40-80 words long. Do not include any explanations, introductions, or meta-commentary. Only output the review text. 
        Enclose your entire output strictly between [[[ and ]]] like this:
        [[[Your review text here.]]]

        Past reviews:
        {context}

        Book: {meta_data['title']}
        Author: {meta_data['author']['name']}
        Summary: {meta_data['book_summary']}

        Review: <|im_end|>
        <|im_start|>assistant /no_think
        """
        promptLog = f"\n\n#### Prompting {self.modelString}: ####\n\n{prompt}\n\n#### End of Prompt ####\n\n"
        logging.debug(promptLog)

        return self.generate(prompt)


    def generate(self, prompt):
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        
        answerLog = f"\n\n#### {self.modelString} Response: ####\n\n{content}\n\n#### End of Response ####\n\n"
        logging.debug(answerLog)

        return content