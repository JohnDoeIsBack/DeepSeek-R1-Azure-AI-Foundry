# Consume DeepSeek R1 with Azure AI Foundry

A python script to get started with DeepSeek R1 on the Azure AI Foundry and working with the models deployed on the microsoft azure cloud.

# Get started in three easy steps:
1. Add your API key
2. Add your Endpoint (Target URI)
3. Modify the user prompt however you need

# Salient features:
* Reasoning and Thinking——the AI model will show its thinking process in real time
* Streaming——the output will be streamed piece by piece so you can see the response generated in real time
* Follows Best practices——by default, the temparature of the model is set at 0 while the suggested value is 0.6. top_p is set at 0.95.
* Readymade User Prompt——an ideal prompt specimen is provided, modify the 'Question' to consume.


# DO NOT add System Prompts 
Adding System Prompts may induce noise and lead to vastly inferior results. DeepSeek works best without it because that's how it was trained. 

Works well with GitHub CodeSpaces. 


**Pre-requisites: pip install azure-ai-inference**
