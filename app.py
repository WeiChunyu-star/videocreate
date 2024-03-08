from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = '/usr/src/app/gemma-2b-it'

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")

@app.route('/invoke', methods=['POST'])
def invoke():
    request_id = request.headers.get("x-fc-request-id", "")
    print("FC Invoke Start RequestId: " + request_id)

    text = request.get_data().decode("utf-8")
    print(text)
    input_ids = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_new_tokens=1000)
    response = tokenizer.decode(outputs[0])
    print("FC Invoke End RequestId: " + request_id)
    return str(response) + "\n"

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=9000)
