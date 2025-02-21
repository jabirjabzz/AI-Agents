#downloading deepseek model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class DeepSeekWrapper:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-1.3b-distilled")
        self.model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-llm-1.3b-distilled",
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate_task(self, user_input):
        prompt = f"User: {user_input}\nAssistant: The browser task should be:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
