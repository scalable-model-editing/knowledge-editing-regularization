## Installation
We work off of the [MEMIT](https://github.com/kmeng01/memit) codebase, so we'll reference the same installation procedures here: 
"We recommend `conda` for managing Python, CUDA, and PyTorch; `pip` is for everything else. To get started, simply install `conda` and run:
```bash
CONDA_HOME=$CONDA_HOME ./scripts/setup_conda.sh
```

`$CONDA_HOME` should be the path to your `conda` installation, e.g., `~/miniconda3`."


## Running the experiments
To evaluate MPES + Norm-Constraint in the code we called ENCORE (Early stopping and Norm-Constrained Robust knowledge Editing), run the following command:

```python
python experiments/evaluate_unified_editing.py \
--alg_name=ENCORE \
--num_edits=100 \
--model_name=gpt2-xl \
--hparams_fname=gpt2-xl.json \
--ds_name=mcf
```

The above script can also be used to run ROME and MEMIT from the same file. We have a common underlying code-base for calculating the key and value vectors. The update equations for ROME, MEMIT and EMMET are in the file unified_editing/unified_main.py 


**Before any experiment is run**, there might be need to update ```sys.path.append('/path/to/encore')``` in the files 'experiments/evaluate_unified_editing.py' and 'experiments/py/eval_utils_zsre.py' 

## Downstream Evaluation

**downstream_tasks** specifies the downstream tasks to run. Available tasks: nli,rte,mrpc,sentiment_analysis,dialogue,nli,cola,sst

**number_of_few_shots** is the number of few shots for each downstream task. Specify the number of few shots for each task, separated by commas. number_of_few_shots must be same length as downstream_tasks. Its default value is 0 when the flag is not provided

**number_of_tests** is the number of tests for all downstream tasks. The default to using the entire test dataset if the flag is not provided

Example:
To run nli, sst and mmlu with 2,3,3 few shots respectively, run the following command:

```python
python experiments/evaluate_unified_editing.py \
--alg_name=ENCORE \
--num_edits=100 \
--model_name=gpt2-xl \
--hparams_fname=gpt2-xl.json \
--ds_name=mcf \
--do_downstream_eval=True \
--downstream_eval_steps=100 \
--downstream_tasks=nli,sst,mmlu,mrpc,cola,rte \
--number_of_few_shots=4,4,4,4,4,4 \
--number_of_tests=100
```

## How to Cite

If you find our work useful, please cite it using the followings:

```
@misc{gupta2025lifelongknowledgeeditingrequires,
      title={Lifelong Knowledge Editing requires Better Regularization},
      author={Akshat Gupta and Phudish Prateepamornkul and Maochuan Lu and Ahmed Alaa and Thomas Hartvigsen and Gopala Anumanchipalli},
      year={2025},
      eprint={2502.01636},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.01636},
}
```