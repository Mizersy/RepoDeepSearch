export PYTHONPATH=$PYTHONPATH:$(pwd)
export PROJECT_FILE_LOC="<PROJECT_FILE_LOC>"

cd ./inference/
pip install -r requirements.txt
pip install vllm==0.6.4
pip install httpx==0.23.3

python -m vllm.entrypoints.openai.api_server     --gpu-memory-utilization 0.95     --served-model-name Qwen-32B     --model Qwen-32B-path --tensor-parallel-size 4 --max-model-len 131072     --trust-remote-code --rope-scaling '{"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}' &
sleep 100

# Patch Generation

# model_name="aws_sdk_claude37_sonnet"
# backend="Anomy"
# threads=10
dataset="princeton-nlp/SWE-bench_Verified"


loc_model_name="ToolTrain-32B"
repair_model_name="Qwen-32B"
backend="openai"
threads=5

python agentless/repair/repair.py --loc_file "<loc_file_path>" \
                                    --output_folder "<output_path>/results/swe-bench-verified/patches_${loc_model_name}_v3_nfc/repair_sample_${repair_model_name}/" \
                                    --loc_interval \
                                    --top_n=5 \
                                    --context_window=10 \
                                    --max_samples 20  \
                                    --cot \
                                    --diff_format \
                                    --function_level \
                                    --gen_and_process \
                                    --dataset "${dataset}" \
                                    --num_threads $threads \
                                    --model "${repair_model_name}" \
                                    --backend "${backend}"

folders=(
    "<output_path>/results/swe-bench-verified/patches_${loc_model_name}_v3_nfc/repair_sample_${repair_model_name}/"
)

for folder in "${folders[@]}"; do
    run_id_prefix=$(basename ${folder%/*})-$(basename $folder)
    python agentless/repair/rerank.py --patch_folder ${folder}  \
                                    --output_file <output_path>/results/swe-bench-verified/patches_${loc_model_name}_v3_nfc/repair_sample_${repair_model_name}/new_${run_id_prefix}.jsonl \
                                    --num_samples 20 \
                                    --deduplicate 
done