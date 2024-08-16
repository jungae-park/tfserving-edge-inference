import os
import numpy as np
import requests
import json
import time
import psutil
import threading
from tensorflow.keras.preprocessing import image as keras_image
from jtop import jtop 

# 모델 서버 URL 설정 
model_server_urls = {
    'Xavier': {
        'MobileNetV1': 'http://xavier_ip:8501/v1/models/mobilenet_v1:predict',
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

# GPU 정보를 저장할 전역 변수
gpu_info = ""
power_info = ""

# GPU 정보 및 전력 정보 측정 함수
def get_gpu_info():
    global gpu_info, power_info
    with jtop() as jetson:
        while jetson.ok():
            # GPU 정보와 전력 소비량 업데이트
            gpu_info = f"GPU Usage: {jetson.stats['GPU']}%"
            power_info = f"Power Draw: {jetson.stats['Power TOT']} mW" # 전체 전력 소비량
            time.sleep(1)  # 1초 간격으로 업데이트

# 이미지 로드 및 전처리 함수
def load_and_preprocess_image(image_path, model_name):
    target_size = (224, 224) if model_name in ['MobileNetV1', 'MobileNetV2'] else (299, 299)
    img = keras_image.load_img(image_path, target_size=target_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# 성능 측정 및 리소스 사용량 출력 함수
def measure_performance(model_server_url, image):
    start_time = time.time()
    
    # 추론 요청 생성
    data = json.dumps({"instances": image.tolist()})
    
    # 추론 요청 전송
    response = requests.post(model_server_url, data=data, headers={"content-type": "application/json"})
    
    if response.status_code != 200:
        raise Exception(f"Request failed with status code: {response.status_code}, response: {response.text}")
    
    predictions = response.json()['predictions']
    
    end_time = time.time()
    
    # 리소스 사용량 측정
    cpu_usage = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    
    # 성능 데이터 반환
    return {
        'inference_time': end_time - start_time,
        'predictions': predictions,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_info.percent,
        'gpu_info': gpu_info,  # 현재 GPU 정보 추가
        'power_info': power_info,  # 현재 전력 정보 추가
    }

# 요청 비율에 따른 작업 수행 함수
def request_distribution_based(device, request_distribution):
    results = []
    total_requests = 10

    for scenario, distribution in request_distribution.items():
        for model_name, percentage in distribution.items():
            num_requests = int(total_requests * (percentage / 100))
            for _ in range(num_requests):
                image_file = np.random.choice(image_files)  # 무작위로 이미지 선택
                image_path = os.path.join(imagenet_path, image_file)
                image = load_and_preprocess_image(image_path, model_name)
                
                # 성능 측정 및 시나리오 정보 추가
                result = measure_performance(model_server_urls[device][model_name], image)
                result['scenario'] = scenario  # 시나리오 정보 추가
                results.append((device, model_name, result))
    
    return results

# 결과를 시나리오 별로 출력하기 위한 함수
def print_results_by_scenario(results):
    scenario_results = {}

    # 시나리오 별로 결과를 분류
    for device, model_name, result in results:
        scenario = result.get('scenario', 'Unknown')

        if scenario not in scenario_results:
            scenario_results[scenario] = []

        scenario_results[scenario].append((device, model_name, result))

    # 시나리오 별로 결과 출력
    for scenario, results in scenario_results.items():
        print(f"\n=== {scenario} ===")
        for i, (device, model_name, result) in enumerate(results):
            print(f"{device} Task {i + 1}: {model_name}")
            print(f"  Inference Time: {result['inference_time']:.4f} seconds")
            print(f"  CPU Usage: {result['cpu_usage']}%")
            print(f"  Memory Usage: {result['memory_usage']}%")
            print(f"  GPU Info: {result['gpu_info']}")
            print(f"  Power Info: {result['power_info']}")  # 전력 정보 출력

# 이미지 경로에서 파일 목록 가져오기
image_files = os.listdir(imagenet_path)

# GPU 정보를 수집하는 스레드 시작
gpu_thread = threading.Thread(target=get_gpu_info)
gpu_thread.daemon = True  # 메인 스레드 종료 시 함께 종료
gpu_thread.start()

# Xavier 장비에서 작업 수행
xavier_results = request_distribution_based('Xavier', request_distribution)

# Nano 장비에서 작업 수행
nano_results = request_distribution_based('Nano', request_distribution)

# 결과 출력
print_results_by_scenario(xavier_results)
print_results_by_scenario(nano_results)
