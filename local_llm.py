#downloading deepseek model
from transformers import AutoTokenizer, AutoModelForCausalLM

class DeepSeekWrapper:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-1.3b-distilled")
        self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-1.3b-distilled")

    def generate_task(self, user_input):
        prompt = f"""User: {user_input}
        Assistant: I need to create a browser task. The task should:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=200)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
