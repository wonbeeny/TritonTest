import sys
sys.path.insert(0, '/opt/tritonserver/backends/python/')

import triton_python_backend_utils as pb_utils
import numpy as np

class TritonPythonModel:
    def initialize(self, properties):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            input_data = input_tensor.as_numpy()
            output_data = input_data * 2  # 간단 로직: 2배
            output_tensor = pb_utils.Tensor("OUTPUT0", output_data)
            responses.append(pb_utils.InferenceResponse([output_tensor]))
        return responses
        
    def finalize(self):
        pass
