import os

import openai

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
    model="gpt-3.5-turbo", prompt="Unterschied zwischen Orkan und Wirbelsturm", temperature=0, max_tokens=2048
)

print(response["choices"][0]["text"])

models = openai.Model.list()


# print all model ids
for model in models.data:
    print(model.id)

# print the first model's id
print(models.data[0].id)

# create a completion
completion = openai.Completion.create(model="ada", prompt="Hello world")

# print the completion
print(completion.choices[0].text)
