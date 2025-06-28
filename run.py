import torch
from transformers import AutoModel

# Load a pretrained checkpoint and move it to the available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(
    "cocom-v0-light-16-mistral-7b", trust_remote_code=True
).to(device)
model.eval()

# Provide a single context and question
contexts = [["Rome is the capital of Italy."]]
questions = ["What is the capital of Italy?"]

# Generate the answer using the helper that prepares inputs automatically
with torch.no_grad():
    answers = model.generate_from_text(
        contexts=contexts, questions=questions, max_new_tokens=32
    )

print("Question:", questions[0])
print("Answer:", answers[0])
