import torch
from transformers import AutoTokenizer
from modeling_cocom import COCOM, COCOMConfig

# Step 1: Define the configuration and initialize the model
config = COCOMConfig(
    decoder_model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    compr_model_name="bert-base-uncased",
    compr_rates=[64],
    compr_linear_type="concat",
    quantization="no",
    generation_top_k=1,
    lora=False,
    attn_implementation=None  # Explicitly disable Flash Attention
)

# Step 2: Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = COCOM(config).to(device)
model.eval()

# Step 3: Load tokenizers
encoder_tokenizer = AutoTokenizer.from_pretrained(config.compr_model_name)
decoder_tokenizer = AutoTokenizer.from_pretrained(config.decoder_model_name)

# Step 4: Prepare input question
question = "What is the capital of Italy?"
enc = encoder_tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(device)
enc_input_ids = enc["input_ids"]
enc_attention_mask = enc["attention_mask"]

# Step 5: Generate answer from the model
with torch.no_grad():
    generated = model.generate(
        enc_input_ids=enc_input_ids,
        enc_attention_mask=enc_attention_mask,
        max_new_tokens=32,
        do_sample=False  # Greedy decoding
    )

# Step 6: Decode and print result
answer = decoder_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
print("Question:", question)
print("Answer:", answer)
