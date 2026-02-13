import tritonclient.http as httpclient
import numpy as np
import time

client = httpclient.InferenceServerClient('localhost:8000', verbose=True)

print('🔥 서버 상태:', client.is_server_ready())
print('🔥 모델 상태:', client.is_model_ready('universal_model', model_version='1'))

# 실제 테스트
input_data = np.array([1.5, 2.5, 3.5], dtype=np.float32)
inputs = [httpclient.InferInput('INPUT0', input_data.shape, 'FP32')]
inputs[0].set_data_from_numpy(input_data)
outputs = [httpclient.InferRequestedOutput('OUTPUT0')]

result = client.infer(
    model_name='universal_model', 
    model_version='1', 
    inputs=inputs, 
    outputs=outputs
    )

print('🎉 입력:', input_data)
print('🎉 출력:', result.as_numpy('OUTPUT0'))  # [[3. 5. 7.]]