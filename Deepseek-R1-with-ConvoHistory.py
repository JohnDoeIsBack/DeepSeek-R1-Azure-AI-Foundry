# pip install azure-ai-inference
import os
import re
import json
from pathlib import Path
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import UserMessage
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
import time

# Set your API key you'll find from the https://ai.azure.com/ portal
api_key = os.getenv("AZURE_INFERENCE_CREDENTIAL", 'Your API key goes here')
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")

# Use the endpoint from your Deepseek model deployment
client = ChatCompletionsClient(
    endpoint='Your Endpoint (Target URI) goes here without the /v1/chat/completions', #Example: https://*.eastus2.models.ai.azure.com
    credential=AzureKeyCredential(api_key)
)

# General purpose User Prompt that can be used for a variety of things.
# Forces the model to start with a thinking phase by prepending "<think>\n".
# DO NOT EVER add a System Prompt to DeepSeek.
# Main prompt template. 
user_prompt = (
    "<think>\n"
    "Instructions:\n"
    "- Adapt response style to match question complexity\n"
    "- Use simple language for casual questions\n"
    "- Provide detailed analysis for complex topics\n"
    "- Include relevant examples when helpful\n"
    "- Express confidence levels when uncertain\n"
    "- Maintain factual accuracy\n"
    "- Use analogies for complex concepts\n"
    "- Format responses for readability\n\n"
    "Role: Knowledgeable assistant with expertise across multiple domains\n"
    "Task parameters:\n"
    "- Style: Natural and adaptive\n"
    "- Depth: Matches question complexity\n"
    "- Format: Clear and readable\n"
    "- Tone: Friendly yet professional\n"
    "- Language: Simple and accessible\n\n"
    """Question: 
    Live Saul Goodman Reaction
    """
)

class ConversationManager:
    def __init__(self, history_file='conversation_history.json', max_history=100):
        self.history_file = Path(history_file)
        self.max_history = max_history
        self.history: List[dict] = self.load_history()
        self.template = self._get_template()
    
    def _get_template(self) -> str:
        return (
            "<think>\n"
            "Instructions:\n"
            "- Adapt response style to match question complexity\n"
            "- Use simple language for casual questions\n"
            "- Provide detailed analysis for complex topics\n"
            "- Include relevant examples when helpful\n"
            "- Express confidence levels when uncertain\n"
            "- Maintain factual accuracy\n"
            "- Use analogies for complex concepts\n"
            "- Format responses for readability\n\n"
            "Role: Knowledgeable assistant with expertise across multiple domains\n"
            "Task parameters:\n"
            "- Style: Natural and adaptive\n"
            "- Depth: Matches question complexity\n"
            "- Format: Clear and readable\n"
            "- Tone: Friendly yet professional\n"
            "- Language: Simple and accessible\n\n"
        )
    
    def load_history(self) -> List[dict]:
        """Load conversation history from JSON file, handling empty or corrupt files"""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    content = f.read().strip()
                    if content:
                        return json.loads(content)
                    return []
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load history from {self.history_file}: {e}")
                return []
        return []
    
    # Add error handling for file operations
    def save_history(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except IOError as e:
            print(f"Failed to save history: {e}")
    
    # Add type validation for prompt and response
    def add_exchange(self, prompt: str, response: str):
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise TypeError("Prompt and response must be strings")
        
        # Extract only the Question part
        question_match = re.search(r'Question:\s*(.+?)(?=\s*$)', prompt, re.DOTALL)
        question_only = question_match.group(1).strip() if question_match else prompt
        
        self.history.append({
            'prompt': question_only,  # Store only the question
            'response': response
        })
        
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        self.save_history()
    
    def get_messages(self) -> List[UserMessage]:
        messages = []
        # Add template only once at the start
        messages.append(UserMessage(content=self.template))
        
        for exchange in self.history:
            # Append just "Question: {stored_question}"
            messages.append(UserMessage(content=f"Question: {exchange['prompt']}"))
            messages.append(UserMessage(content=exchange['response']))
        return messages

# Replace conversation_history with ConversationManager
conversation_manager = ConversationManager()

# Remove token limits comment
model_info = client.get_model_info()
print("Model name:", model_info.model_name)
print("Model type:", model_info.model_type)
print("Model provider name:", model_info.model_provider_name)

# Clean up make_api_call function
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=10))
def make_api_call(client, messages):
    """Execute API call with retries"""
    try:
        response = client.complete(
            messages=messages,
            temperature=0.6,
            top_p=0.95,
            max_tokens=4096,
            stream=True
        )
        return response
    except Exception as e:
        print(f"\nAPI call failed: {e}")
        raise

# Remove the direct printing in print_stream function
def print_stream(result):
    """Streams raw response from the model"""
    full_text = ""
    final_update = None
    
    try:
        for update in result:
            final_update = update  # Track final update for usage stats
            if not update or not update.choices:
                continue
            
            try:
                delta = update.choices[0].delta.content
                if delta:
                    full_text += delta
                    print(delta, end="", flush=True)
            except AttributeError:
                continue
                
        # Print usage stats at end of stream
        print("\n\nUsage Statistics:")
        try:
            print(f"\tPrompt tokens: {final_update.usage.prompt_tokens}")
            print(f"\tCompletion tokens: {final_update.usage.completion_tokens}")
            print(f"\tTotal tokens: {final_update.usage.total_tokens}")
        except AttributeError:
            print("\tNot available in streaming response")
                
    except Exception as e:
        print(f"\nStream error: {e}")
        return None

    return full_text if full_text else None

# Update main execution
result = make_api_call(client, conversation_manager.get_messages() + [UserMessage(content=user_prompt)])
if not result:
    print("Failed to get response from API")
    exit(1)

full_response = print_stream(result)
if not full_response:
    print("Failed to process response stream")
    exit(1)

print()  # Newline after streaming

# Still needed for conversation history
match = re.match(r"<think>(.*?)</think>(.*)", full_response, re.DOTALL)
conversation_manager.add_exchange(user_prompt, full_response)
