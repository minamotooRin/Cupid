# 查看支持的任务数据集
lm-eval --tasks list

TASK=mmlu_anatomy,mmlu_clinical_knowledge,mmlu_college_medicine,mmlu_medical_genetics,mmlu_professional_medicine
TASK=multimedqa

# 加载huggingface远程模型
lm_eval --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B,trust_remote_code=true \
    --tasks $TASK \
    --device cuda:0 \
    --batch_size auto:1 \
    --output_path ./eval_out/llama3_8B \
    --use_cache ./eval_cache

# 加载本地模型
CONF=conf2
lm_eval --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B,peft=./output/$CONF/checkpoint-1 \
    --tasks $TASK \
    --device cuda:0 \
    --batch_size auto \
    --output_path ./eval_out/$CONF
    # --use_cache ./eval_cache

