ENGINE_DIR=/engines/gpt/fp16/1-gpu
TOKENIZER_DIR=/root/TensorRT-LLM/examples/gpt/gpt2
MODEL_FOLDER=/triton_model_repo
TRITON_MAX_BATCH_SIZE=16
INSTANCE_COUNT=1
MAX_QUEUE_DELAY_MS=50000
MAX_QUEUE_SIZE=0
FILL_TEMPLATE_SCRIPT=/root/tensorrtllm_backend/tools/fill_template.py
DECOUPLED_MODE=false

mkdir -p ${MODEL_FOLDER}
cp -r /root/tensorrtllm_backend/all_models/inflight_batcher_llm/* ${MODEL_FOLDER}/

python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/ensemble/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},logits_datatype:TYPE_FP32
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_backend:python,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching,max_queue_size:${MAX_QUEUE_SIZE},encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32,exclude_input_in_output:true
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT},max_queue_size:${MAX_QUEUE_SIZE}


# python3 scripts/launch_triton_server.py --model_repo=/triton_model_repo