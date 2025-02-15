# AgentDropout



## Qucik Start

**Create and install the environment**

```shell
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
```

**Set url and API keys in** `AgentDropout/llm/gpt_chat`

```python
MINE_BASE_URL = ""
MINE_API_KEYS = ""
```

**Or use vllm for local deployment**

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/model --dtype auto --api-key API_KEYS --port 6789
```

```python
api_key = API_KEYS
base_url = "http://localhost:6789/v1"
```

**Download Datasets**

Download MMLU, AQuA, MultiArith, SVAMP, HumanEval and GSM8K datasets from [Huggingface]((https://huggingface.co/)). And put them in `datasets`.

**Run AgentDropout**

```shell
python experiments/run_mmlu.py --agent_nums 5 --mode Random --batch_size 40 --num_iterations 10 --imp_per_iterations 5 --pruning_rate 0.20 --num_rounds 2 --llm_name /path/to/model --optimized_spatial --optimized_temporal --diff --dec

python experiments/run_gsm8k.py --agent_nums 5 --mode Random --batch_size 40 --num_iterations 2 --imp_per_iterations 1 --pruning_rate 0.20 --num_rounds 2 --llm_name /path/to/model --optimized_spatial --optimized_temporal --diff --dec

python experiments/run_aqua.py --agent_nums 5 --mode Random --batch_size 40 --num_iterations 2 --imp_per_iterations 1 --pruning_rate 0.20 --num_rounds 2 --llm_name /path/to/model --optimized_spatial --optimized_temporal --diff --dec

python experiments/run_multiarith.py --agent_nums 5 --mode Random --batch_size 60 --num_iterations 2 --imp_per_iterations 1 --pruning_rate 0.20 --num_rounds 2 --llm_name /path/to/model --optimized_spatial --optimized_temporal --diff --dec

python experiments/run_svamp.py --agent_nums 5 --mode Random --batch_size 60 --num_iterations 2 --imp_per_iterations 1 --pruning_rate 0.20 --num_rounds 2 --llm_name /path/to/model --optimized_spatial --optimized_temporal --diff --dec

python experiments/run_humaneval.py --agent_nums 5 --mode Random --batch_size 10 --num_iterations 10 --imp_per_iterations 5 --pruning_rate 0.20 --num_rounds 4 --llm_name /path/to/model --optimized_spatial --optimized_temporal --diff --dec
```



