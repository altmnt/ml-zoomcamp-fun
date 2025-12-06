FROM agrigorev/model-2025-hairstyle:v1

RUN pip install --no-cache-dir \
    onnx onnxscript onnxruntime \
    pillow numpy torchvision

COPY function.py ./lambda_function.py 

CMD ["lambda_function.lambda_handler"]
