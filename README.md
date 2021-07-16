---
language: sv
widget:
- text: "Jag är en svensk språkmodell."
---

# GPT2-svenska-wikipedia
A swedish GPT2 style model trained using Flax CLM pipeline on the Swedish
part of the wiki40b dataset.

https://huggingface.co/datasets/wiki40b


## Data cleaning and preprocessing
The data was cleaned and preprocessed using the following script. Make sure to install depencies for beam_runner to make the dataset work.

```python
from datasets import load_dataset
def load_and_clean_wiki():
    dataset = load_dataset('wiki40b', 'sv', beam_runner='DirectRunner', split="train")
    #dataset = load_dataset('wiki40b', 'sv', beam_runner='DirectRunner')
    dataset = dataset.remove_columns(['wikidata_id', 'version_id'])
    filtered_dataset = dataset.map(filter_wikipedia)
    # filtered_dataset[:3]
    # print(filtered_dataset[:3])
    return filtered_dataset

def filter_wikipedia(batch):
    batch["text"] = " ".join(batch["text"].split("\n_START_SECTION_\n"))
    batch["text"] = " ".join(batch["text"].split("\n_START_ARTICLE_\n"))
    batch["text"] = " ".join(batch["text"].split("\n_START_ARTICLE_\n"))
    batch["text"] = " ".join(batch["text"].split("\n_START_PARAGRAPH_\n"))
    batch["text"] = " ".join(batch["text"].split("_NEWLINE_"))
    batch["text"] = " ".join(batch["text"].split("\xa0"))
    return batch
```

## Training script
The following training script was used to train the model.
```bash
./run_clm_flax.py     --output_dir="${MODEL_DIR}"     --model_type="gpt2"     --config_name="${MODEL_DIR}"     --tokenizer_name="${MODEL_DIR}"     --dataset_name="wiki40b"     --dataset_config_name="sv"     --do_train --do_eval     --block_size="512"     --per_device_train_batch_size="64"     --per_device_eval_batch_size="64"     --learning_rate="5e-3" --warmup_steps="1000"     --adam_beta1="0.9" --adam_beta2="0.98" --weight_decay="0.01"     --overwrite_output_dir     --num_train_epochs="20"     --logging_steps="500"     --save_steps="1000"     --eval_steps="2500"     --push_to_hub
```

