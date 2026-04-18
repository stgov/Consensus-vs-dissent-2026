# Consensus-vs-dissent-2026
Repository for the paper: Consensus vs. Dissent: Dynamic LLM Modeling of Subjective Preferences in Group Recommenders



## Content:

strats: functions to calculate social choice-based aggregation strategies 

config_gen: functions to generate group configurations (uniform, divergent, coalitional, minority)

basemodels: generate LLM-as-judge assessments using base models (using either Cerebras or Ollama) -> but easily adjustable for use with e.g. Huggingface

train_v3_3.jsonl: augmented training data used for fine-tuning

test_v5_2.jsonl: test data for fine-tuning


## LLM-as-judge prompts

```
system_message = {
        'role': 'system',
        'content': """
        You are a participant in a scientific study. 
        You will be presented with four scenarios related to four different groups of people. 
        For each of them, a software system will produce recommendations on the basis of the preferences of the group members. 
        Please read carefully the description of each scenario, and then answer the question. 

        The statements are to be answered using a 7-point Likert scale:
        -3: strongly disagree
        -2: disagree
        -1: somehwat disagree
        0: neither agree, neither disagree
        1: somewhat agree
        2: agree
        3: strongly agree

        You are to return ONLY a valid JSON object, with no explanations or extra text. 
        Do not include markdown formatting (no ```json). 
        The JSON must have the following structure:
        {{
        "fairness": your reply,
        }}
                """
                }
```

```
scenario = {
        'role': 'user',
        'content': f"""
        Assume that there is a group of friends. Every month, a group decision is made by these friends to decide on a restaurant to have dinner together. 
        To select a restaurant for the dinner next month, the group again has to take the same decision. 
        In this decision, each group member explicitly rated ten possible restaurants using a 5-star rating scale (1: the worst, 5: the best). 
        The ratings given by group members are shown in the table below:

        ### BEGIN TABLE ###
        {result}
        ### END TABLE ###

        The group decided to avoid going in the same restaurant too often; hence, after a restaurant has been selected, it cannot be chosen again for the next 4 dinners. 
        **The last 3 restaurants visited are: {already_visited} **.
        Using the provided ratings, the system made a suggestion for the group on the basis of the preferences of the all the group members.
        **{recommendation}** has been recommended to the group {explanation}.

        Indicate your agreement to the following statements (7-point likert scale ranging from -3 to 3):

        ** Fairness **
        The group recommendation is fair to all group members.


        Reply only with the json object which includes your reply (integer 7-point likert scale -3 to 3). 
        """

    }
```

## Fine-tuning Documentation

### Llama

Llama was fine-tuned using Supervised fine-tuning (SFT) with the transformers python package for four epochs. Data was split 80/20 training/test. The test set was used for evaluation during fine-tuning. Below is the training and evaluation loss.

![Llama-loss](Llama-training-loss.png)

#### Parameters:
```
lr:  1e-4
lora_alpha:32
lora_dropout:0.1
lora_rank:16
epochs: 4
per_device_eval_batch_size:2
per_device_train_batch_size:4
grad_accum: 4
max_seq_length: 1275 (calculated based on max length of samples)
```


### Olmo3

Llama was fine-tuned using Supervised fine-tuning (SFT) with the transformers python package for four epochs. Data was split 80/20 training/test. The test set was used for evaluation during fine-tuning. Below is the training and evaluation loss.

![Olmo-loss](OLMO_loss.png)

#### Parameters:
```
lr:  2e-5
lora_alpha:32
lora_dropout:0.1
lora_rank:16
epochs: 4
per_device_eval_batch_size:2
per_device_train_batch_size:4
grad_accum: 4
max_seq_length: 1275 (calculated based on max length of samples)
```

Custom masking function (create_weighted_labels(input_ids, tokenizer, model_name)) to only calculate loss on assistant message, which was validated before running. Function created to also parse Olmo and Qwen tokenizers
e.g.,

Unmasked: 271 of a total of 1086 tokens
