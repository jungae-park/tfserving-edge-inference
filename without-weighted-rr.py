import os
import numpy as np
import requests
import json
import time
import psutil
from tensorflow.keras.preprocessing import image as keras_image

# 모델 서버 URL 설정 
model_server_urls = {
    'Xavier': {
        'MobileNetV1': 'http://xavier_ip:8501/v1/models/mobilenet_v1:predict',
        'MobileNetV2': 'http://xavier_ip:8501/v1/models/mobilenet_v2:predict',
        'InceptionV3': 'http://xavier_ip:8501/v1/models/inception_v3:predict'
    },
    'Nano': {
        'MobileNetV1': 'http://nano_ip:8501/v1/models/mobilenet_v1:predict',
        'MobileNetV2': 'http://nano_ip:8501/v1/models/mobilenet_v2:predict',
        'InceptionV3': 'http://nano_ip:8501/v1/models/inception_v3:predict'
    }
}

# 이미지 경로 설정
imagenet_path = './dataset/imagenet/imagenet_1000_raw'

# 이미지 로드 및 전처리 함수
def load_and_preprocess_image(image_path, model_name):
    if model_name in ['MobileNetV1', 'MobileNetV2']:
        target_size = (224, 224)
    elif model_name == 'InceptionV3':
        target_size = (299, 299)
    else:
        raise ValueError("Unknown model name")
    
    img = keras_image.load_img(image_path, target_size=target_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# 성능 측정 및 리소스 사용량 출력 함수
def measure_performance(model_server_url, image):
    start_time = time.time()
    
    # 추론 요청 생성
    data = json.dumps({
        "instances": image.tolist()
    })
    
    # 추론 요청 전송
    response = requests.post(model_server_url, data=data, headers={"content-type": "application/json"})
    
    if response.status_code != 200:
        raise Exception(f"Request failed with status code: {response.status_code}, response: {response.text}")
    
    predictions = response.json()['predictions']
    
    end_time = time.time()
    
    # 리소스 사용량 측정
    cpu_usage = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    #gpu_usage = get_gpu_usage()
    
    # 성능 데이터 반환
    return {
        'inference_time': end_time - start_time,
        'predictions': predictions,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_info.percent,
    #    'gpu_usage': gpu_usage
    }

# GPU 사용량 측정 함수
def get_gpu_usage():
    try:
        # tegrastats 명령어 실행
        result = os.popen("tegrastats | tail -n 1").read()
        # 결과에서 GPU 사용량 추출
        for line in result.split():
            if 'GR3D' in line: 
                gpu_usage = line.split('%')[0].strip()
                return int(gpu_usage)
    except Exception as e:
        gpu_usage = None
        print(f"Error retrieving GPU usage: {e}")
    return gpu_usage

# RR 알고리즘에 따른 작업 분배
def round_robin(tasks, device):
    results = []
    for model_name in tasks:
        # 이미지 로드
        image_file = np.random.choice(image_files)  # 무작위로 이미지 선택
        image_path = os.path.join(imagenet_path, image_file)
        image = load_and_preprocess_image(image_path, model_name)
        
        # 성능 측정
        result = measure_performance(model_server_urls[device][model_name], image)
        results.append((model_name, result))
    return results

# 10개의 작업을 설정
tasks = ['MobileNetV1', 'MobileNetV2', 'InceptionV3'] * 3 + ['MobileNetV1']

# ImageNet 데이터셋에서 이미지를 무작위로 선택
image_files = os.listdir(imagenet_path)

# Xavier 장비에서 RR 알고리즘을 사용하여 작업 수행
print("Processing on Xavier:")
results_xavier = round_robin(tasks, 'Xavier')

# 결과 출력 (Xavier)
for i, (model_name, result) in enumerate(results_xavier):
    print(f"Task {i + 1}: {model_name} on Xavier")
    print(f"  Inference Time: {result['inference_time']:.4f} seconds")
    print(f"  CPU Usage: {result['cpu_usage']}%")
    print(f"  Memory Usage: {result['memory_usage']}%")
#    print(f"  GPU Usage: {result['gpu_usage']}%")

# Nano 장비에서 RR 알고리즘을 사용하여 작업 수행
print("\nProcessing on Nano:")
results_nano = round_robin(tasks, 'Nano')

# 결과 출력 (Nano)
for i, (model_name, result) in enumerate(results_nano):
    print(f"Task {i + 1}: {model_name} on Nano")
    print(f"  Inference Time: {result['inference_time']:.4f} seconds")
    print(f"  CPU Usage: {result['cpu_usage']}%")
    print(f"  Memory Usage: {result['memory_usage']}%")
#    print(f"  GPU Usage: {result['gpu_usage']}%")
