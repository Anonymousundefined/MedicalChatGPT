import torch
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, BitsAndBytesConfig


class LLama:
    def __init__(self, base_model, lora_weights):
        self.config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        self.lora_model = lora_weights
        self.basemodel = base_model
        self.model = LlamaForCausalLM.from_pretrained(self.basemodel,
                                                      load_in_4bit=True,
                                                      torch_dtype=torch.float16,
                                                      quantization_config=self.config,
                                                      device_map="auto")
        self.tokenizer = LlamaTokenizer.from_pretrained(self.basemodel)
        self.model = PeftModel.from_pretrained(self.model, self.lora_model)

        self.generation_config = GenerationConfig(
            temperature=0.01,
            top_p=0.9,
            typical_p=0.9,
            repetition_penalty=5.0,
            encoder_repetition_penalty=5.0,
            top_k=40,
            renormalize_logits=True,
            do_sample=True,
            num_beams=2,
            num_return_sequences=1,
            remove_invalid_values=True
        )

    def response(self, query, input=None):
        PROMPT = f"""If you are a doctor, please answer the medical questions based on the patient's description.

              ### Input:
              {query}


              ### Response: """
        inputs = self.tokenizer(
            PROMPT,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].cuda()

        generation_output = self.model.generate(
            input_ids=input_ids,
            generation_config=self.generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=256,
        )

        response = generation_output.sequences[0]
        return self.process_response(self.tokenizer.decode(response))

    def process_response(self, text):
        """
        Process the text to give response only.

        Args:
            text (str): The input text containing the response.

        Returns:
            str: The extracted text.
        """
        start = text.find("Response:") + len("Response:")
        text = text[start:].strip()
        first_chat_doctor = text.find("Chat Doctor.")
        second_chat_doctor = text.find("Chat Doctor.", first_chat_doctor + len("Chat Doctor."))

        if second_chat_doctor != -1:
            result_text = text[:second_chat_doctor]
        else:
            result_text = text

        return result_text
