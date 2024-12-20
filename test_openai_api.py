from openai import OpenAI

client = OpenAI(api_key="your-api-key")

# Replace with your actual OpenAI API key

try:
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ])

    print("Response:")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"An error occurred: {e}")
