# CINE Frame Extractor

## 소개
CINE Frame Extractor는 고속 카메라로 촬영된 CINE 파일에서 프레임을 추출하고 처리하는 Python 애플리케이션입니다. 이 도구는 PySide6를 사용한 GUI 인터페이스를 제공하며, 멀티프로세싱을 활용하여 빠른 프레임 처리 속도를 제공합니다.

## 주요 기능
- CINE 파일에서 프레임 추출
- 프레임 그룹화 및 시간 정보 추가
- 멀티프로세싱을 통한 고속 처리
- 사용자 지정 시작 프레임, 프레임 수, 스트라이드 설정
- 처리 진행 상황 실시간 표시
- CSV 로그 파일 생성

## 요구 사항
- pip install -r requirements.txt
- Python 3.8+
- PySide6
- OpenCV
- NumPy
- pycine

## 사용 방법
1. 애플리케이션을 실행합니다:
2. GUI에서 CINE 파일을 선택합니다.
3. 출력 디렉토리를 선택합니다.
4. 시작 프레임, 프레임 수, 스트라이드를 설정합니다.
5. "Extract Frames" 버튼을 클릭하여 프레임 추출을 시작합니다.

## 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다.
