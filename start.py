import ollama


response = ollama.list()
# print(response)

res = ollama.chat(
    model= 'llama3.2',
    messages = [
        {"role": "user", "content": "Why is the sky blue?"}
    ],
    stream=True
)

for chunk in res:
    print(chunk["message"]["content"], end="", flush=True)