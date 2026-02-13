import sys
sys.path.insert(0, '/opt/tritonserver/backends/python/')

import json
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        # 필요하면 초기화 로직 작성
        pass

    def execute(self, requests):
        responses = []

        for request in requests:
            # 1) 입력 텐서 가져오기 (TYPE_STRING, dims [1])
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_JSON")
            # Triton의 TYPE_STRING → numpy object array of bytes 로 옴
            in_data = in_tensor.as_numpy()  # shape: (batch, 1)

            batch_out = []

            # bytes → str
            raw_str = in_data[0].decode("utf-8")
            # str(JSON) → dict
            in_dict = json.loads(raw_str)

            # 여기서 원하는 로직 수행 (예: 키 추가)
            out_dict = {
                "received": in_dict,
                "message": "processed by TritonPythonModel"
            }

            # dict → JSON 문자열 → bytes
            out_str = json.dumps(out_dict, ensure_ascii=False)
            batch_out.append(out_str.encode("utf-8"))

            # 2) 출력 numpy array 생성 (object dtype, shape [batch, 1])
            out_np = np.array(batch_out, dtype=object).reshape(in_data.shape[0], 1)

            out_tensor = pb_utils.Tensor("OUTPUT_JSON", out_np)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        # 리소스 정리 필요 시 사용
        pass