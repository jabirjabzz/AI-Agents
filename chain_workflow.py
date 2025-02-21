from browser_use import Agent
from local_llm import DeepSeekWrapper
import asyncio
import torch
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ChainedAgent:
    def __init__(self, headless: bool = True):
        """
        Initialize the chained agent with local LLM and browser automation
        
        Args:
            headless (bool): Run browser in headless mode (True) or visible (False)
        """
        self.logger = logging.getLogger(__name__)
        self.llm = DeepSeekWrapper()
        self.browser_agent = Agent(
            llm=None,  # We handle LLM separately
            headless=headless,
            timeout=30  # Seconds before timing out browser operations
        )
        self.conversation_history = []
        self.max_history_length = 5  # Keep last 5 exchanges

    async def execute_chain(self, user_input: str) -> str:
        """
        Execute full chain: User Input → LLM Task Gen → Browser Execution
        
        Args:
            user_input (str): Natural language request from user
            
        Returns:
            str: Result of browser automation or error message
        """
        try:
            # Validate input safety
            if not self._is_input_safe(user_input):
                return "Input contains blocked patterns - request denied"

            # Store user input with timestamp
            self._update_history(f"User: {user_input}")
            
            # Generate browser task with local LLM
            raw_task = self.llm.generate_task(user_input)
            self.logger.debug(f"Raw generated task: {raw_task}")
            
            # Clean and validate the task
            cleaned_task = self._clean_task(raw_task)
            if not self._validate_task(cleaned_task):
                return "Generated task failed validation checks"

            # Execute browser automation
            self.browser_agent.task = cleaned_task
            result = await self.browser_agent.run()
            
            # Store and return result
            self._update_history(f"Assistant: {result}")
            return result

        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            self.logger.error(error_msg)
            self._update_history(f"System Error: {error_msg}")
            return error_msg

    def _is_input_safe(self, text: str) -> bool:
        """Check for prohibited patterns in user input"""
        blocked_patterns = [
            "password", "credit card", "delete", 
            "sudo", "admin", "http://", "https://"
        ]
        return not any(pattern in text.lower() for pattern in blocked_patterns)

    def _clean_task(self, raw_task: str) -> str:
        """Extract only the relevant task instructions"""
        # Remove any code blocks
        clean_task = raw_task.split("```")[0]
        # Remove LLM response artifacts
        for phrase in ["Assistant:", "Here's the task:", "Task:"]:
            clean_task = clean_task.replace(phrase, "")
        return clean_task.strip()

    def _validate_task(self, task: str) -> bool:
        """Ensure generated task is executable"""
        required_verbs = ["go to", "click", "search", "open", "return"]
        return any(verb in task.lower() for verb in required_verbs)

    def _update_history(self, entry: str):
        """Manage conversation history with rolling window"""
        self.conversation_history.append(entry)
        # Keep only recent history
        self.conversation_history = self.conversation_history[-self.max_history_length:]

async def main():
    agent = ChainedAgent(headless=False)  # Set headless=True for production
    
    while True:
        user_input = input("\nEnter your request (or 'exit'): ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        result = await agent.execute_chain(user_input)
        print(f"\nResult: {result}")

    print("\nSession history:")
    for entry in agent.conversation_history:
        print(f"- {entry}")

if __name__ == "__main__":
    asyncio.run(main())