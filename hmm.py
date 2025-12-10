from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

max_memory = {
    0: "12GiB",   # use up to 12 GB on GPU0 for weights
    1: "12GiB",   # use up to 12 GB on GPU1 for weights
    "cpu": "64GiB",  # offload the rest to CPU if needed
}

olmo = AutoModelForCausalLM.from_pretrained(
    #"./models/olmo3-7b-instruct",
    "allenai/Olmo-3-7B-Instruct",
    max_memory=max_memory,
    device_map="auto"
)


olmo = AutoModelForCausalLM.from_pretrained(
    "allenai/Olmo-3-7B-Instruct", 
    torch_dtype=torch.float16, 
    load_in_8bit=True)

olmo.config.use_cache = False
#olmo = olmo.to('cuda')

tokenizer = AutoTokenizer.from_pretrained(
    #"./models/olmo3-7b-instruct",
    "allenai/Olmo-3-7B-Instruct")

message = ["What is soup really?"]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
# optional verifying cuda
inputs = {k: v.to(olmo.device) for k,v in inputs.items()}
response = olmo.generate(**inputs, max_new_tokens=32768, temperature=0.6, top_p=0.95)
print(tokenizer.batch_decode(response, skip_special_tokens=True))

torch.cuda.empty_cache()

eos_token_ids = olmo.generation_config.eos_token_id
print("EOS token ids:", eos_token_ids)
print("Last token:", response[0, -1].item())
print("Ended by EOS?", response[0, -1].item() in (eos_token_ids if isinstance(eos_token_ids, list) else [eos_token_ids]))

print(tokenizer.batch_decode([100265, 100257], skip_special_tokens=True))
