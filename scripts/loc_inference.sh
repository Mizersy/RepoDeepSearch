export PYTHONPATH=$PYTHONPATH:$(pwd)
export PROJECT_FILE_LOC="<PROJECT_FILE_LOC>"
cd ./inference/
pip install -r requirements.txt


pip install -r ../verl/requirements_sglang.txt
pip install vllm==0.6.4
pip install httpx==0.23.3


# qwen
models=("ToolTrain-32B")
model_names=("ToolTrain-32B")
backend=("openai")
threads=10


python -m vllm.entrypoints.openai.api_server     --gpu-memory-utilization 0.95     --served-model-name ToolTrain-32B     --model ToolTrain-32B-path --tensor-parallel-size 4 --max-model-len 32768     --trust-remote-code &
sleep 100
echo "server started...."
for i in "${!models[@]}"; do
  python RepoSearcher/fl/AFL_localize_file.py --file_level \
                               --output_folder "<output_path>/results/swe-bench-verified/file_level_${models[$i]}_v3_nfc" \
                               --num_threads ${threads} \
                               --model "${model_names[$i]}" \
                               --backend "${backend[$i]}" \
                               --dataset "princeton-nlp/SWE-bench_Verified" \
                               --skip_existing \
                               --native_function_call

done


for i in "${!models[@]}"; do
  python RepoSearcher/fl/AFL_localize_func.py \
    --output_folder "<output_path>/results/swe-bench-verified/func_level_${models[$i]}_v3_nfc" \
    --loc_file "<output_path>/results/swe-bench-verified/file_level_${models[$i]}_v3_nfc/loc_outputs.jsonl" \
    --output_file "loc_${models[$i]}_func.jsonl" \
    --temperature 0.0 \
    --model "${model_names[$i]}" \
    --backend "${backend[$i]}" \
    --dataset "princeton-nlp/SWE-bench_Verified" \
    --skip_existing \
    --num_threads ${threads} \
    --native_function_call
done