import onnxruntime as ort
from io import BytesIO
from urllib import request
from PIL import Image
from torchvision import transforms

prepare_image = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ) # ImageNet normalization
])

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


session = ort.InferenceSession(
    "hair_classifier_empty.onnx", providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def predict(url):
    X = prepare_image(download_image(url))
    result = session.run([output_name], {input_name: X.unsqueeze(0).numpy()})
    float_predictions = result[0][0].tolist()
    return float_predictions


def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result