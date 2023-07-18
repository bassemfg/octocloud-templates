import torch
from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt-large")
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt-large")
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

result=generator("COVID-19 is", max_length=20, num_return_sequences=5, do_sample=True)
print(result)


sentence = "COVID-19 is"
inputs = tokenizer(sentence, return_tensors="pt")


with torch.no_grad():
    beam_output = model.generate(**inputs,
                                min_length=100,
                                max_length=1024,
                                num_beams=5,
                                early_stopping=True
                                )
result = tokenizer.decode(beam_output[0], skip_special_tokens=True)
print(result)
'COVID-19 is a global pandemic caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), the causative agent of coronavirus disease 2019 (COVID-19), which has spread to more than 200 countries and territories, including the United States (US), Canada, Australia, New Zealand, the United Kingdom (UK), and the United States of America (USA), as of March 11, 2020, with more than 800,000 confirmed cases and more than 800,000 deaths.'
