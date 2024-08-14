import os
import numpy as np
import requests
import json
import time
import psutil
import subprocess

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

# 요청 비율 설정
request_distribution = {
    'Scenario1': {'MobileNetV1': 50.0, 'MobileNetV2': 30.0, 'InceptionV3': 20.0},
    'Scenario2': {'MobileNetV2': 60.0, 'InceptionV3': 40.0},
    'Scenario3': {'MobileNetV1': 40.0, 'MobileNetV2': 30.0, 'InceptionV3': 30.0},
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
    gpu_info = get_gpu_info()  # GPU 사용량 및 전력 정보 측정
    
    # 성능 데이터 반환
    return {
        'inference_time': end_time - start_time,
        'predictions': predictions,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_info.percent,
        'gpu_info': gpu_info,
    }

# GPU 사용량 및 전력 정보 측정 함수 (tegrastats 사용)
def get_gpu_info():
    try:
        # tegrastats를 백그라운드로 실행
        process = subprocess.Popen(['tegrastats'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(1)  # 1초 동안 대기하여 정보 수집
        output, errors = process.communicate(timeout=1)  # 출력 가져오기
        power_info = output.decode('utf-8').strip().split('\n')
        
        # 필요한 정보 필터링
        gpu_info = []
        for line in power_info:
            if 'GR3D' in line:  # GPU 정보
                gpu_info.append(line)
        
        return "\n".join(gpu_info) if gpu_info else "GPU 정보가 없습니다."
    except Exception as e:
        return f"GPU 정보를 가져오는 중 오류 발생: {e}"

# 배터리 정보 측정 함수 (Jetson 보드는 일반적으로 배터리 정보를 제공하지 않음)
def get_battery_info():
    try:
        # tegrastats를 백그라운드로 실행
        process = subprocess.Popen(['tegrastats'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(1)  # 1초 동안 대기하여 정보 수집
        output, errors = process.communicate(timeout=1)  # 출력 가져오기
        power_info = output.decode('utf-8').strip().split('\n')
        
        # 필요한 정보 필터링
        battery_info = []
        for line in power_info:
            if 'POM_5V' in line:  # 배터리 정보
                battery_info.append(line)
        
        return "\n".join(battery_info) if battery_info else "배터리 정보가 없습니다."
    except Exception as e:
        return f"배터리 정보를 가져오는 중 오류 발생: {e}"

# 요청 비율에 따른 작업 수행 함수
def request_distribution_based(device, tasks, request_distribution):
    results = []
    total_requests = 10

    for scenario, distribution in request_distribution.items():
        for model_name, percentage in distribution.items():
            num_requests = int(total_requests * (percentage / 100))
            for _ in range(num_requests):
                image_file = np.random.choice(image_files)  # 무작위로 이미지 선택
                image_path = os.path.join(imagenet_path, image_file)
                image = load_and_preprocess_image(image_path, model_name)
                
                # 성능 측정
                result = measure_performance(model_server_urls[device][model_name], image)
                results.append((device, model_name, result))
    
    return results

# 모델 목록 설정
tasks = ['MobileNetV1', 'MobileNetV2', 'InceptionV3']

# ImageNet 데이터셋에서 이미지를 무작위로 선택
image_files = os.listdir(imagenet_path)

# Xavier 장비에서 작업 수행
xavier_results = request_distribution_based('Xavier', tasks, request_distribution)

# Nano 장비에서 작업 수행
nano_results = request_distribution_based('Nano', tasks, request_distribution)

# 결과 출력
for i, (device, model_name, result) in enumerate(xavier_results):
    print(f"Xavier Task {i + 1}: {model_name} on {device}")
    print(f"  Inference Time: {result['inference_time']:.4f} seconds")
    print(f"  CPU Usage: {result['cpu_usage']}%")
    print(f"  Memory Usage: {result['memory_usage']}%")
    print(f"  GPU Info: {result['gpu_info']}")
    print(f"  Battery Info: {get_battery_info()}")  # 배터리 정보 추가 출력

for i, (device, model_name, result) in enumerate(nano_results):
    print(f"Nano Task {i + 1}: {model_name} on {device}")
    print(f"  Inference Time: {result['inference_time']:.4f} seconds")
    print(f"  CPU Usage: {result['cpu_usage']}%")
    print(f"  Memory Usage: {result['memory_usage']}%")
    print(f"  GPU Info: {result['gpu_info']}")
    print(f"  Battery Info: {get_battery_info()}")  # 배터리 정보 추가 출력
