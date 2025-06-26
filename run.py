import torch
from modeling_cocom import COCOM, COCOMConfig

# Step 1: Define the configuration and initialize the model
config = COCOMConfig(
    decoder_model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    compr_model_name="bert-base-uncased",
    compr_rates=[64],
    compr_linear_type="concat",
    quantization="no",
    generation_top_k=1,
    lora=False
)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Modify the configuration to disable flash attention if running on CPU
if device.type == "cpu":
    config.attn_implementation = None  # Disable flash attention

# Initialize the model and move it to the appropriate device
model = COCOM(config).to(device)

# Step 2: Prepare dummy input data
batch_size = 2
enc_input_ids = torch.randint(0, 1000, (batch_size, 128)).to(device)
enc_attention_mask = torch.ones_like(enc_input_ids).to(device)
dec_input_ids = torch.randint(0, 1000, (batch_size, 64)).to(device)
dec_attention_mask = torch.ones_like(dec_input_ids).to(device)
labels = torch.randint(0, 1000, (batch_size, 64)).to(device)

# Step 3: Test the forward pass
output = model(
    enc_input_ids=enc_input_ids,
    enc_attention_mask=enc_attention_mask,
    dec_input_ids=dec_input_ids,
    dec_attention_mask=dec_attention_mask,
    labels=labels
)
print("Loss:", output["loss"])
print("Logits shape:", output["logits"].shape)