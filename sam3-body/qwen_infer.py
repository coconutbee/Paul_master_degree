from transformers import pipeline

pipe = pipeline("image-text-to-text", model="Qwen/Qwen3.5-4B")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
result = pipe(text=messages)
# 將結果存入變數並印出
assistant_response = result[0]['generated_text'][-1]['content']

print(assistant_response)