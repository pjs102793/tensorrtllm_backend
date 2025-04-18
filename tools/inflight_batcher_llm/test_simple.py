import os
import sys

import numpy as np
from datetime import datetime
import multiprocessing
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import utils

def simple_inference_test(request_id=1):
    # 기본 설정
    protocol = 'http'
    url = 'localhost:8000'
    model_name = 'tensorrt_llm'
    concurrency = 128

    # 클라이언트 생성
    print(f"[INFO] [{request_id}] 클라이언트 생성 중 ({protocol}, {url})...")
    try:
        client = utils.create_inference_server_client(
            protocol, url, concurrency=concurrency, verbose=False)
    except Exception as e:
        print(f"[ERROR] [{request_id}] 채널 생성 실패: {str(e)}")
        return

    np.random.seed(int(time.time()) + request_id)  # 또는 hash(request_id) 등

    # 입력 데이터 준비 (128 토큰 길이)
    input_length = 450
    input_ids = np.random.randint(0, 30000, size=(1, input_length), dtype=np.int32)
    input_lengths = np.array([[input_length]], dtype=np.int32)
    output_len = np.array([[450]], dtype=np.int32)  # 20 토큰 생성
    pad_id = np.array([[0]], dtype=np.int32)
    end_id = np.array([[1]], dtype=np.int32)

    # 텐서 생성
    inputs = [
        utils.prepare_tensor("input_ids", input_ids, protocol),
        utils.prepare_tensor("input_lengths", input_lengths, protocol),
        utils.prepare_tensor("request_output_len", output_len, protocol),
        utils.prepare_tensor("pad_id", pad_id, protocol),
        utils.prepare_tensor("end_id", end_id, protocol),
    ]

    # 추론 요청
    print(f"[INFO] [{request_id}] 추론 요청 시작...")
    start_time = datetime.now()

    result = client.infer(model_name, inputs, request_id=str(request_id))

    # 결과 출력
    output_ids = result.as_numpy("output_ids")
    seq_lengths = result.as_numpy("sequence_length")

    stop_time = datetime.now()
    latency = (stop_time - start_time).total_seconds() * 1000.0
    latency = round(latency, 3)

    print(f"[INFO] [{request_id}] 추론 완료: 지연 시간 {latency} ms")
    print(f"[INFO] [{request_id}] 출력 ID 형태: {output_ids.shape}")
    print(f"[INFO] [{request_id}] 시퀀스 길이: {seq_lengths}")
    print(f"[INFO] [{request_id}] 첫 10개 출력 토큰: {output_ids[0, 0, :10]}")

def run_parallel_inference(num_processes, num_batches=1, delay_between_batches=0):
    """여러 프로세스로 추론을 병렬 실행합니다."""
    total_requests = num_processes * num_batches
    print(f"[INFO] 총 {total_requests}개 요청 병렬 처리 시작 (프로세스: {num_processes}, 배치: {num_batches})")

    start_time = datetime.now()

    for batch in range(num_batches):
        processes = []
        for i in range(num_processes):
            request_id = batch * num_processes + i + 1
            p = multiprocessing.Process(target=simple_inference_test, args=(request_id,))
            processes.append(p)
            p.start()

        # 모든 프로세스 완료 대기
        for p in processes:
            p.join()

        if batch < num_batches - 1 and delay_between_batches > 0:
            print(f"[INFO] 다음 배치 전 {delay_between_batches}초 대기 중...")
            time.sleep(delay_between_batches)

    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    print(f"[INFO] 모든 요청 완료. 총 실행 시간: {total_time:.2f}초")
    print(f"[INFO] 평균 처리량: {total_requests / total_time:.2f} 요청/초")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TensorRT-LLM 추론 병렬 테스트')
    parser.add_argument('--processes', type=int, default=16, help='동시에 실행할 프로세스 수 (기본값: 32)')
    parser.add_argument('--batches', type=int, default=1, help='실행할 배치 수 (기본값: 1)')
    parser.add_argument('--delay', type=float, default=0, help='배치 간 지연 시간(초) (기본값: 0)')

    args = parser.parse_args()

    if args.processes == 1 and args.batches == 1:
        simple_inference_test()
    else:
        run_parallel_inference(args.processes, args.batches, args.delay)