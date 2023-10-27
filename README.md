# xDial-Eval
Repository for EMNLP-2023 Findings Paper - xDial-Eval: A Multilingual Open-Domain Dialogue Evaluation Benchmark

## Changelog

[25/10/2023] Add data to the repository.

[27/10/2023] Add code for zero-shot inference with open-source LLMs to the repository.

## Prerequisites

- Python 3.8+ and PyTorch 1.13.1+
- See requirments.txt

## Data Format

1. The csv files in the turn-level data include columns: ``[lang]_ctx``, ``[lang]_res``, and ``raings`` where ``[lang]`` refers to different languages.
2. The csv files in the dialogue-level data include columns: ``[lang]_dial`` and ``raings`` where ``[lang]`` refers to different languages.
3. ``[lang]_ctx`` and ``[lang]_dialogue`` are delimited by ```\n```.

## Original English Data

### Sources
- FED-Turn and FED-Dial: http://shikib.com/fed_data.json
- Persona-USR and Topical-USR: http://shikib.com/usr
- DailyDialog-Zhao and Persona-Zhao: https://github.com/ZHAOTING/dialog-processing/tree/master/src/tasks/response_eval
- ConTurE-Dial and ConTurE-Turn: https://github.com/alexa/conture
- Empathetic-GRADE, ConvAI2-GRADE, and DailyDialog-GRADE: https://github.com/li3cmz/GRADE/tree/main/evaluation
- Persona-DSTC10 and Topical-DSTC10: https://chateval.org/dstc10
- DailyDialog-Gupta: https://github.com/prakharguptaz/multirefeval
- IEval: https://github.com/Sea94/ieval
- Persona-See: https://github.com/facebookresearch/ParlAI/tree/main/projects/controllable_dialogue
- Reliable-Eval: https://github.com/TianboJi/Dialogue-Eval
- Human-Eval: https://github.com/facebookresearch/ParlAI/tree/main/projects/humaneval 

Note that for accessing Human-Eval data, please contact the original authors of [Human Evaluation of Conversations is an Open Problem: comparing the sensitivity of various methods for evaluating dialogue agents](https://aclanthology.org/2022.nlp4convai-1.8/). Once you have obtained the permission, you may contact me to obtain the multilingual extension of Human-Eval data.

### Acknowledge Statement
We thank all the authors for kindly making their data publicly available. In the same spirit, we make our multilingual extension publicly available as well. We hope our data can further benefit researchers working on multilingual open-domain dialogue systems and evaluation metrics.

## Zero-shot Inference with Open-source LLMs

Currently, I included scripts for zero-shot inference with LLama-2, Baichuan-2, Phoenix, and Alpaca. You can easily adapt the scripts to other open-source LLMs. 

The python scripts can be found in ``zeroshot_inference`` and the shell scripts are in ``scripts/zeroshot_inference``. 

Example execution - ``bash zeroshot_inference/turn/infer_alpaca.sh``. 

## Code for Finetuning Open-source LLMs (Coming Soon)
