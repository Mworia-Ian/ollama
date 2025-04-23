import ollama

res = ollama.generate(
    model="llama3.2",
    prompt="why is the sky blue"
)