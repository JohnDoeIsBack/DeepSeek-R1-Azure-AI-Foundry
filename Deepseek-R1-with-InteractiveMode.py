import os
import re
import sys
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import UserMessage
from typing import Optional, Dict, Any

class DeepseekAI:
    def __init__(self):
        self.api_key = os.getenv("AZURE_INFERENCE_CREDENTIAL", 'Your API Key goes here')
        if not self.api_key:
            raise ValueError("API key not found. Please set your API key")
            
        self.client = ChatCompletionsClient(
            endpoint='Your Endpoint goes here without the /v1/chat/completions', #Example: https://*.eastus2.models.ai.azure.com
            credential=AzureKeyCredential(self.api_key)
        )
        
        # Store template separately for reuse
        self.template = self._get_template()
        self.conversation_history = []
        
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
    
    def print_model_info(self):
        """Display model information with error handling"""
        try:
            model_info = self.client.get_model_info()
            print("\nModel Information:")
            print(f"Name: {model_info.model_name}")
            print(f"Type: {model_info.model_type}")
            print(f"Provider: {model_info.model_provider_name}\n")
        except Exception as e:
            print(f"Warning: Could not get model info: {e}\n")

    def print_stream(self, result) -> str:
        """Enhanced streaming output with formatting"""
        full_text = ""
        try:
            for update in result:
                if update.choices:
                    delta = update.choices[0].delta.content
                    
                    # Format different content types
                    if delta.strip().startswith(('#', '-', '*', '1.', '```', '>')):
                        print('\n', end='', flush=True)
                        if delta.strip().startswith('```'):
                            print('\n', end='', flush=True)
                    
                    full_text += delta
                    print(delta, end="", flush=True)
        except Exception as e:
            print(f"\nError during streaming: {e}")
        return full_text

    def process_response(self, response: str):
        """Process and format the AI response"""
        match = re.match(r"<think>(.*?)</think>(.*)", response, re.DOTALL)
        
        print("\nResponse:")
        if match:
            print("\nThinking Process:")
            print(match.group(1).strip())
            print("\nAnswer:")
            print(match.group(2).strip())
        else:
            print(response.strip())

    def get_completion(self, question: str) -> Optional[str]:
        """Get completion with error handling"""
        # Build context from previous exchanges
        messages = [
            UserMessage(content=exchange) 
            for exchange in self.conversation_history
        ]
        # Add new question
        messages.append(UserMessage(content=self.template + "Question: " + question))
        
        try:
            result = self.client.complete(messages=messages)  # Sends all history
            self.conversation_history.append(question)
            self.conversation_history.append(result)
            return self.print_stream(result)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return None

    def print_usage(self, result):
        """Print token usage statistics"""
        try:
            print("\nToken Usage:")
            print(f"Prompt tokens: {result.usage.prompt_tokens}")
            print(f"Completion tokens: {result.usage.completion_tokens}")
            print(f"Total tokens: {result.usage.total_tokens}")
        except AttributeError:
            pass

    def interactive_session(self):
        """Run an interactive session"""
        print("\n=== DeepSeek AI Interactive Mode ===")
        print("Type 'exit' to quit, 'clear' to clear screen")
        self.print_model_info()
        
        while True:
            try:
                question = input("\nQuestion: ").strip()
                
                if question.lower() == 'exit':
                    print("\nGoodbye!")
                    break
                elif question.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                elif not question:
                    continue
                
                response = self.get_completion(question)
                if response:
                    self.process_response(response)
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)

def main():
    try:
        ai = DeepseekAI()
        ai.interactive_session()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()