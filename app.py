from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

@app.route("/api/chat", methods=["POST"])
def chat():
    message = request.json["message"]

    # 在这里执行ChatGPT的回复生成逻辑
    reply = generate_reply(message)

    return jsonify({"reply": reply})

def generate_reply(message):
    # 在这里使用ChatGPT模型生成回复
    # 这里使用了Hugging Face的transformers库，您需要根据自己的模型和配置进行相应的修改
    model_name = "gpt2"  # 使用的预训练模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    inputs = tokenizer.encode(message, return_tensors="pt")
    reply_ids = model.generate(inputs, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    return reply

if __name__ == "__main__":
    app.run()
