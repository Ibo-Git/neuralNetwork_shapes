from transformers import GPTJForCausalLM, AutoTokenizer
import torch
import os
import time


## CUSTOM USER INPUT
prompt = "Ich bin Markus Wiese und mir geht es "
temperature = 0.9
max_length = 100
## END CUSTOM USER INPUT






# Load cached model+tokenizer
load_start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelpath = os.path.join(os.getcwd(), "saved_files\\gptj",  'gptj-model.pth')

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = torch.load(modelpath, map_location=torch.device(device))
print("Load completed in - ", time.time() - load_start_time)
# If model is not yet saved - execute this code instead of above code
#model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B",  torch_dtype=torch.float16)
#model = model.to(device)
#torch.save(model, modelpath)

generate_start_time = time.time()
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
gen_tokens = model.generate(input_ids, do_sample=True, temperature=temperature, max_length=max_length)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print("Generation completed in - ", time.time() - generate_start_time)
print("--------------------------------")
print(gen_text)
print("--------------------------------")