# GPT-2-Haiku
My minimal-ish port of GPT-2 to JAX/haiku. I'd recommend the HuggingFace FLAX one instead, as that was released after I initially ported this from the OpenAI implementation.

# Running
* Install JAX, and the requirements:
```
pip install -r requirements.txt
```
* Download the weights [here]()

* Choose a text file, and run the training script:
```
python train.py my_text_file.py --init_path=models/1558M --out_path=my_saved_model
```
