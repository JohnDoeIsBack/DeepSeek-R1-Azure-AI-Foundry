# Consume DeepSeek R1 with Azure AI Foundry

An Easy Peasy python script to get started with DeepSeek R1 on the Azure AI Foundry and working with the models deployed on the microsoft azure cloud.

# Get started in three easy steps:
1. Add your API key
2. Add your Endpoint (Target URI)
3. Modify the user prompt however you need

# Salient features:
* Reasoning and Thinking——the AI model will show its thinking process in real time
* Streaming——the output will be streamed piece by piece so you can see the response generated in real time
* Follows Best practices——by default, the temperature of the model is set at 0 while the suggested value is 0.6. top_p is set at 0.95.
* Readymade User Prompt——an ideal prompt specimen is provided, modify the 'Question' to consume.
* Minimal changes needed——nothing apart from the 'user_prompt' needs any changes. The ideal settings are already set for 99% use cases.

# Instructions:
1. To simply get started with one-shot prompting, use _Deepseek-R1-with-AzureAI.py_. This is for beginners and easiest to get started.

2. To store ephemeral context and interactivity like a Chatbot, use _Deepseek-R1-with-InteractiveMode.py_. The 'conversation_history' will remember your previous prompts (questions) and the responses from the  AI model as long as the loop continues. Type 'exit' to stop the interaction.

3. **((Best))** For persistent conversation History, use _Deepseek-R1-with-ConvoHistory.py_. All the previous prompts and responses will be stored in JSON format and passed to the next prompt. You only need to modify the 'user_prompt' every time.

PSA: The history will be deleted in a FILO order after the 'max_history' reaches 100. This is to prevent the input token from overbloating. You can increase or decrease this value however you like.

PSA: Max tokens are set at 4096 to prevent an unsuspecting user from overconsuming the API and avoiding extraordinary charges. DeepSeek-R1 supports an input limit of 16,384 tokens and an output limit of 163,840 tokens on Azure.

# DO NOT add System Prompts 
Adding System Prompts may induce noise and lead to vastly inferior results. DeepSeek works best without it because that's how it was trained. 

Works well with GitHub CodeSpaces. 


**Pre-requisites: pip install azure-ai-inference**
