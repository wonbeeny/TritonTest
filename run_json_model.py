# 개략 예시 (HTTP client)
import json
import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient("localhost:8000")

payload = {"foo": 1, "bar": "baz"}
payload = ex_input
payload_str = json.dumps(payload)
input_np = np.array([payload_str.encode("utf-8")], dtype=object)

inp = httpclient.InferInput("INPUT_JSON", input_np.shape, "BYTES")
inp.set_data_from_numpy(input_np)

out = httpclient.InferRequestedOutput("OUTPUT_JSON")
res = client.infer(
    model_name="json_model", 
    model_version='1',
    inputs=[inp], 
    outputs=[out]
    )

out_np = res.as_numpy("OUTPUT_JSON")
out_str = out_np[0][0].decode("utf-8")
out_dict = json.loads(out_str)
print('🎉 입력:\n', input_np)
print('\n\n🎉 출력\n:', out_dict)