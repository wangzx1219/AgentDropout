# AgentDropout

### About Method

We propose **AgentDropout**, a novel topology optimization method for Multi-agent system with domain transferability and structure robustness. AgentDropout dynamically adjusts the participating agents and communication links among agents in each round, allowing for more flexible and adaptive team configurations. 

<img src="image/README/main.png" alt="main" style="zoom: 33%;" />

### File Structure

| Directory       | Contents              |
| --------------- | --------------------- |
| `datasets/`     | Experimental data     |
| `AgentDropout/` | Main codes            |
| `experiments/`  | Test scripts          |
| `result/`       | Few samples of output |

### Requirements

Install anaconda environment

```shell
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
```

Set url and API keys in `AgentDropout/llm/gpt_chat`

```python
MINE_BASE_URL = ""
MINE_API_KEYS = ""
```

Or use vllm for local deployment

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/model --dtype auto --api-key API_KEYS --port 6789
```

```python
api_key = API_KEYS
base_url = "http://localhost:6789/v1"
```

Prepare data from [Huggingface](https://huggingface.co/). And put them in `datasets/`.

### Quick Start

Run AgentDropout on GSM8K, the same as other datasets: 

```shell
python experiments/run_gsm8k.py --agent_nums 5 --mode FullConnected --batch_size 40 --num_iterations 2 --imp_per_iterations 1 --pruning_rate 0.20 --num_rounds 2 --llm_name /data/models/Meta-Llama-3-8B-Instruct --optimized_spatial --optimized_temporal --diff --dec
```

Code framework based on [GPTSwarm](https://github.com/metauto-ai/GPTSwarm) and [AgentPrune](https://github.com/yanweiyue/AgentPrune).
