# nano-llm
A custom implementation of a custom large language model from scratch in PyTorch. Includes training and post-training processes.

## Table of Contents
- [nano-llm](#nano-llm)
  - [Table of Contents](#table-of-contents)
  - [Architecture](#architecture)
    - [Multi-Head Latent Attention](#multi-head-latent-attention)
    - [Mixture of Experts](#mixture-of-experts)
  - [Training](#training)
  - [Post-Training](#post-training)
  - [Installation](#installation)
  - [Usage](#usage)
  - [License](#license)

## Architecture
The architecture of NanoLLM was mainly inspired by [Llama4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/),  [DeepSeekV3](https://arxiv.org/abs/2412.19437) and [Andrej Karpathy]()'s open-source projects.
TODO
### Multi-Head Latent Attention 
TODO
### Mixture of Experts
TODO
[SwiGLU](https://arxiv.org/pdf/2002.05202v1)
Its main advantage is that it provides a smoother transition around 0, which leads to better optimization and faster convergence.

## Training 
Datasets to be used: OpenWebText, SuperNaturalInstruction, OpenAssistant, GRK8K, commonquery

## Post-Training
TODO group relative policy optimization

## Installation
1. Clone the repository:
```bash
git clone https://github.com/asoloveii/nano-llm.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
 ```

3. (Optional) Run code from scratch yourself
```bash
commands to download datasets TODO
```

## Usage
huggingface links...

## License
This project is licensed under the [MIT License](LICENSE).
