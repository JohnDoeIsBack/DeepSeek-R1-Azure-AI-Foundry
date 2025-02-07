# pip install azure-ai-inference
import os
import re
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import UserMessage

# Recommended configurations for DeepSeek-R1 series:
# - Temperature: 0.6 (range 0.5-0.7 is advised)
# - Top_p: 0.95 for nucleus sampling
# - Do not use a system prompt; include all instructions within the user prompt.
# - To enforce thorough reasoning, prepend "<think>\n" to the prompt.
# - For math problems, consider adding: "Please reason step by step, and put your final answer within \boxed{}."
# - Generate multiple completions (64) to estimate pass@1 by averaging results.

# Set your API key you'll find from the https://ai.azure.com/ portal
api_key = os.getenv("AZURE_INFERENCE_CREDENTIAL", 'Your API Key goes here')
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")

# Use the endpoint from your Deepseek model deployment
client = ChatCompletionsClient(
    endpoint='Your Endpoint goes here without the /v1/chat/completions', #Example: https://*.eastus2.models.ai.azure.com
    credential=AzureKeyCredential(api_key)
)

# Get and display information about the AI model
model_info = client.get_model_info()
print("Model name:", model_info.model_name)
print("Model type:", model_info.model_type)
print("Model provider name:", model_info.model_provider_name)

# Create a user prompt that contains all instructions (no system prompt)
# and forces the model to start with a thinking phase by prepending "<think>\n".
# Main prompt template - tells the AI how to behave and respond
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
    # Below is the actual prompt to ask the AI model. Multiline quote so you can expand the prompt without any worries.
    """Question: 
    Why are we still here? Just to suffer?
    """
)

# Coding exclusive user prompt; uncomment to use. Do not use two 'user_prompt' variables simultaneously.
"""
user_prompt = (
    "<think>\n"
    "Instructions:\n"
    "- Provide production-ready code examples\n"
    "- Include error handling and edge cases\n"
    "- Show design patterns and best practices\n"
    "- Explain time/space complexity\n"
    "- Include unit test examples\n"
    "- Reference language-specific documentation\n"
    "- Highlight potential security concerns\n"
    "- Show performance optimization tips\n\n"
    "Role: Expert Software Architect specializing in system design and optimization\n"
    "Task parameters:\n"
    "- Code quality: Production-ready\n"
    "- Documentation: Comprehensive\n"
    "- Standards: Industry best practices\n"
    "- Security: OWASP compliance\n\n"
    "Question: [Your coding question here]\n"
)
"""

# Use streaming with the recommended parameters.
result = client.complete(
    messages=[
        UserMessage(content=user_prompt)
    ],
    temperature=0.6,   # Recommended value to balance randomness and coherence.
    top_p=0.95,        # Nucleus sampling with a probability mass of 0.95.
    max_tokens=4096,
    stream=True,       # Use streaming to get partial results piece by piece
    #model_extras={"n": 64}  # Generate 64 completions to estimate pass@1.
)

# Helper function to display streaming responses
def print_stream(result):
    """
    Prints the chat completion with streaming and returns the full accumulated content.
    """
    full_text = ""
    for update in result:
        if update.choices:
            # Use the lowercase 'content' attribute.
            delta = update.choices[0].delta.content
            full_text += delta
            print(delta, end="")
    return full_text

# Capture and print the streaming output.
full_response = print_stream(result)
print()  # Newline after streaming

# Look for any "thinking" section in the response
# The AI might explain its thought process in <think> tags
match = re.match(r"<think>(.*?)</think>(.*)", full_response, re.DOTALL)

print("Response:")
if match:
    print("\tThinking:", match.group(1))
    print("\tAnswer:", match.group(2))
else:
    print("\tAnswer:", full_response)

# Integrated snippet: print model and usage details, if available.
try:
    print("Model:", result.model)
    print("Usage:")
    print("\tPrompt tokens:", result.usage.prompt_tokens)
    print("\tTotal tokens:", result.usage.total_tokens)
    print("\tCompletion tokens:", result.usage.completion_tokens)
except AttributeError:
    # Skip printing if these details are not provided by the streaming API.
    pass
