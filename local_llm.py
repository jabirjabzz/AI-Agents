from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

class DeepSeekWrapper:
    def __init__(self):
        model_name = "microsoft/phi-3-mini-4k-instruct"
        
        # 4-bit quantization for low memory usage
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=quantization_config
        )

    def generate_task(self, user_input):
        prompt = f"""<|user|>
        Create a browser automation task for: {user_input}
        <|assistant|>"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)