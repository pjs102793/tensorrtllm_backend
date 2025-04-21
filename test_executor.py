import tensorrt_llm.bindings.executor as trtllm

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executor Bindings Example")
    parser.add_argument("--model_path",
                        type=str,
                        required=True,
                        help="Directory containing model engine")
    args = parser.parse_args()

    kv_cache_config = trtllm.KvCacheConfig(enable_block_reuse=False)

    # Create the executor.
    executor = trtllm.Executor(args.model_path, trtllm.ModelType.DECODER_ONLY, trtllm.ExecutorConfig(kv_cache_config=kv_cache_config))

    if executor.can_enqueue_requests():
        # Create the request.
        request = trtllm.Request(input_token_ids=[1, 2, 3, 4, 5], max_tokens=30)
        request2 = trtllm.Request(input_token_ids=[1, 2, 3], max_tokens=50)
        request3 = trtllm.Request(input_token_ids=[1, 2, 3, 4], max_tokens=20)
        request4 = trtllm.Request(input_token_ids=[1, 2, 3, 4, 5], max_tokens=40)
        request5 = trtllm.Request(input_token_ids=[1, 2, 3, 4, 5], max_tokens=90)
        
        requests = [request, request2, request3, request4, request5]

        # Enqueue the request.
        request_ids = executor.enqueue_requests(requests)

        # Wait for the new tokens
        received = 0
        all_responses = []

        while received < len(request_ids):
            responses = executor.await_responses()
            for resp in responses:
                print(f"[INFO] Request {resp.request_id} completed")
                print(f"[INFO] Output tokens: {resp.result.output_token_ids}")
                all_responses.append(resp)
                received += 1
                
            # for stat in executor.get_latest_iteration_stats():
            #     value = stat.inflight_batching_stats.num_scheduled_requests
            #     print(value)
