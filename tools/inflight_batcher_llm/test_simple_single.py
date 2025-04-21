import os
import sys
import numpy as np
from datetime import datetime
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import utils  # 반드시 이 경로에 utils가 있어야 합니다

def run_single_inference(request_id=1):
    protocol = 'http'
    url = 'localhost:8000'
    model_name = 'tensorrt_llm'

    print(f"[INFO] [{request_id}] 클라이언트 생성 중 ({protocol}, {url})...")
    try:
        client = utils.create_inference_server_client(
            protocol, url, concurrency=1, verbose=False)
    except Exception as e:
        print(f"[ERROR] [{request_id}] 클라이언트 생성 실패: {str(e)}")
        return

    # 입력 데이터 구성
    input_length = 128
    input_ids = np.random.randint(0, 30000, size=(1, input_length), dtype=np.int32)
    input_lengths = np.array([[input_length]], dtype=np.int32)
    output_len = np.array([[128]], dtype=np.int32)
    pad_id = np.array([[0]], dtype=np.int32)
    end_id = np.array([[1]], dtype=np.int32)

    inputs = [
        utils.prepare_tensor("input_ids", input_ids, protocol),
        utils.prepare_tensor("input_lengths", input_lengths, protocol),
        utils.prepare_tensor("request_output_len", output_len, protocol),
        utils.prepare_tensor("pad_id", pad_id, protocol),
        utils.prepare_tensor("end_id", end_id, protocol),
    ]
    
    print(input_ids)
    print(input_lengths)
    print(output_len)

    print(f"[INFO] [{request_id}] 추론 요청 시작...")
    start_time = datetime.now()

    for _ in range(10):
        result = client.infer(model_name, inputs, request_id=str(request_id))

    output_ids = result.as_numpy("output_ids")
    seq_lengths = result.as_numpy("sequence_length")
    context_logits = result.as_numpy("context_logits")

    latency = (datetime.now() - start_time).total_seconds() * 1000.0
    print(f"[INFO] [{request_id}] 추론 완료: 지연 시간 {round(latency, 2)} ms")
    print(f"[INFO] [{request_id}] 출력 ID 형태: {output_ids.shape}")
    print(f"[INFO] [{request_id}] 시퀀스 길이: {seq_lengths}")
    print(f"[INFO] [{request_id}] 생성 로그 확률: {context_logits}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='단일 추론 테스트')
    parser.add_argument('--request_id', type=int, default=1)
    args = parser.parse_args()

    run_single_inference(args.request_id)