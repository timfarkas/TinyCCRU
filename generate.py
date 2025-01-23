import torch
from train_transformer import BigramLanguageModel, decode

# Load the model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = BigramLanguageModel()
model.load_state_dict(torch.load("final_model.pth", map_location=device))
model = model.to(device)
model.eval()

## wait between generations if on GPU 
wait = True if device == "cuda" else False

# Generate tokens indefinitely and print in real time
context = torch.zeros((1, 1), dtype=torch.long, device=device)

import time

while True:
    generated_tokens = model.generate(context, max_new_tokens=5)[0].tolist()
    generated_text = decode(generated_tokens)
    print(generated_text[-5:], end='', flush=True)
    
    context = torch.tensor([generated_tokens], dtype=torch.long, device=device)
    
    if wait:
        time.sleep(0.05) 