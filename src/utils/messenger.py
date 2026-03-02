import os
import requests
from dotenv import load_dotenv


def send_message(dim, overlap, method, lambda_val, best_P_succ, theory_P_succ, avg_lag, time, trial, sim_filename):
    load_dotenv()
    uri = os.getenv("MESSENGER_URI")
    print(f"INFO: 대상 URI: {uri}")

    body_dict = {
        "botName": "VQSD OAM 실험 결과 알림",
        "botIconImage": "https://static.dooray.com/static_images/dooray-bot.png",
        "text": f"🚀 VQSD 실험 완료\n"
                f"* 설정: Dim {dim}, Overlap {overlap}, method {method}, lambda {lambda_val}\n"
                f"* 결과: \n"
                f"\t* Best success rate: {best_P_succ:.4f} (Theory: {theory_P_succ:.4f})\n"
                f"\t* Avg lag: {avg_lag:.4f}\n"
                f"* 시간: {time} (Trials: {trial})\n"
                f"* 파일명: {sim_filename}"
    }

    try:
        response = requests.post(
            uri,
            json=body_dict
        )

        if response.status_code == 200:
            print("SUCCESS: 메시지가 성공적으로 전송되었습니다.")
        else:
            print(f"ERROR: 요청 실패. HTTP 상태 코드: {response.status_code}")
            print(f"응답 본문: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"FATAL ERROR: 요청 중 오류 발생 - {e}")