# Finetuning starter kit

This code is built on HuggingFace Transformers, Datasets, and Models, and allows you to do any combination of:
- training/finetuning any causal LM from the HF Model Hub on the HackerCup dataset
- evaluating on the same
- sampling n Python code generations from the model checkpoint
- running all generations on the sample input/output pairs, and choosing the highest scoring one to run the full input/output pairs

*WARNING*: This code will run code generated by an LLM on your machine without checking it for safety. DO THIS AT YOUR OWN RISK. By default, this script will terminate after program generation, to give you time to read the LLM generations. We recommend that you manually check all LLM-generated code before running it. Options to disable this behavior are described below.

## Installation

Install requirements with
```
pip install -r requirements.txt
```
An environment manager (e.g. conda) is recommended. Depending on your exact CUDA version and other factors, you may need to install some requirements manually.

## Training and evaluation

### To evaluate a pretrained model on HackerCup by generating n solutions and selecting the best

```
python train_and_eval.py --output_dir ~/temp --num_gens 10 --max_code_gen_examples -1   ## Does not run generated code
## We highly recommend you read the generated code here before proceeding.
python train_and_eval.py --output_dir ~/temp --only_run_generated_code 
```
`--output_dir` is required but is not used if not finetuning.

This command will currently generate code, but will *not* run it immediately. You will need to run the second command above in order to run and evaluate this code on the test set inputs. You can turn off this behavior and trigger the entire process with only one command thus, but again, we recommend against this unless you know what you're doing:

```
python train_and_eval.py --output_dir ~/temp --num_gens 10 --max_code_gen_examples -1  --run_generated_code 
```

We currently filter to only include problems from HackerCup for which each test case occupies
exactly one input and one output line, as we are only providing a starting point. This significantly
reduces the number of available examples, from 284 to 17. However, models can handle input and output parsing and formatting, and this starter kit does this parsing programmatically only because it is meant to serve as a starting point.

Command-line arguments controlling some data and model hyperparameters are defined at the top of
`train_and_eval.py`. The most relevant may include:
-  `--text_columns`, the set of columns included in the input to the model.
-  `--num_gens`, the number of generations to sample for each problem.
-  `--max_code_gen_examples`, the number of HackerCup problems to sample code generations for.
-  `--model_name_or_path`, the HuggingFace model to use. 

Beware that the problem statements in the HackerCup dataset are quite long. For many models using fixed-max-length positional embeddings (this excludes, e.g., RoPE), including the problem statement as-is will significantly exceed the context length.

### To finetune a pretrained model on HackerCup dataset and evaluate as above

```
python train_and_eval.py --do_train --do_eval --output_dir <OUTPUT_DIR> --num_gens 10
```
This command trains on the HackerCup dataset using a causal language modeling objective, by which we mean
that the model is trained to maximize the probability of the next token, and that the loss is applied to
all tokens. See the next section if you want a loss only on the code solution.

`--do_eval` also triggers evaluation on the training dataset, as a causal LM task. The HackerCup dataset
does not come with a precomputed train/val/test split, so one is created in the code. This may be non-deterministic.

Additionally, because we are using the same HackerCup dataset for training and the final code generation eval,
we *are* training on the test set in this demo, albeit in a more roundabout way, since the metrics are not directly computed on the code generations. To adapt this code to tune on other tasks instead, see below.

Command-line arguments controlling some data and model hyperparameters are defined at the top of
`train_and_eval.py`. We also use the HuggingFace `transformer.TrainingArguments` hyperparameters. A 
[list of options](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments)
including train epochs, batch size, learning rate, and precision can be found in the transformers documentation.


### To train only on code (or another part of the problems)

It is possible to only apply LM loss to tokens in the code solutions by using a
[`trl.SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer#trl.SFTTrainer).

This behavior is triggered with `--train_obj seq2seq`. It is currently implemented in `train_hf_model()` and depends on the presence of the string `### Code:`, which we manually add in each example (see `format_example()`). If you make changes to the code, e.g. to allow for code generation in other languages, as below, it's possible you'll break this behavior.

### To adapt this code to generate code in C/C++ (or any other language)

The example code solutions for the HackerCup dataset are primarily written in C++, but we opt to generate
Python code in this kit.

To generate other coding languages, changes to `format_datasets()` are required,
including adding a "C" or "C++" case for the if-then statements, to check for the `language` argument.
The `template` string also needs to be rewritten for this language, though
your model's generation should be capable of handling input and output parsing/formatting.

### To adapt this code for finetuning on other HF datasets

The dataset to train on is determined by `--dataset_name`.

Modifications need to be made to `format_datasets()`, which is currently written
specifically to ingest examples in the HackerCup format. For some datasets, this can
simply return the `text` field of an example.

Some command line arguments, such as `--text_columns` need to be adjusted for the columns in your dataset.