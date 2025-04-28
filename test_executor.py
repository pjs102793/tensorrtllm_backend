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
 
    # SamplingConfig(
    #     end_id=8193,
    #     pad_id=8807,
    #     max_new_tokens=200, -> max_tokens
    #     num_beams=1,
    #     output_sequence_lengths=True,
    #     temperature=0.8,
    #     top_k=50,
    #     top_p=0.8,
    #     length_penalty=1.0,
    #     early_stopping=1, -> use beam search
    #     repetition_penalty=2.0,
    #     min_length=1,
    #     random_seed=None,
    # )
    sampling_kwargs = dict(
        beam_width=1,
        top_k=50,
        top_p=0.8,
        random_seed=42,
        temperature=0.8,
        repetition_penalty=2.0,
        length_penalty=1.0,
        num_return_sequences=1,
        min_length=1,
    )
    # None 제거 (깔끔하게)
    sampling_kwargs = {k: v for k, v in sampling_kwargs.items() if v is not None}

    sampling_config = trtllm.SamplingConfig(**sampling_kwargs)

    # Create the executor.
    executor = trtllm.Executor(args.model_path, trtllm.ModelType.DECODER_ONLY, trtllm.ExecutorConfig(kv_cache_config=kv_cache_config))

    if executor.can_enqueue_requests():
        # Create the request.
        request = trtllm.Request(input_token_ids=[1, 2, 8193], max_tokens=50, end_id=8193, pad_id=8807, sampling_config=sampling_config)

        requests = [request, request, request, request, request]

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
