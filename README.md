# CINE Frame Extractor

## 소개
CINE Frame Extractor는 고속 카메라로 촬영된 CINE 파일에서 프레임을 추출하고 처리하는 Python 애플리케이션입니다. 이 도구는 PySide6를 사용한 GUI 인터페이스를 제공하며, 멀티스레딩을 활용하여 빠른 프레임 처리 속도를 제공합니다.

## 주요 기능
- CINE 파일에서 프레임 추출
- 프레임 그룹화 및 시간 정보 추가
- 멀티스레딩을 통한 고속 처리
- 사용자 지정 시작 프레임, 프레임 수, 스트라이드 설정
- 처리 진행 상황 실시간 표시
- CSV 로그 파일 생성

## 요구 사항
- Python 3.8+
- PySide6
- OpenCV
- NumPy
- pycine

## 설치
1. 저장소를 클론합니다:
2. 필요한 패키지를 설치합니다: pip install -r requirements.txt

## 사용 방법
1. 애플리케이션을 실행합니다: dist\CINE_Frame_Extractor.exe
2. GUI에서 CINE 파일을 선택합니다.
3. 출력 디렉토리를 선택합니다.
4. 시작 프레임, 프레임 수, 스트라이드를 설정합니다.
5. "Extract Frames" 버튼을 클릭하여 프레임 추출을 시작합니다.

## 윈도우 실행 파일 빌드
1. PyInstaller 설치: pip install pyinstaller
2. spec 파일 생성: pyi-makespec gui.py --name "CINE_Frame_Extractor" --windowed --onefile
3. spec 파일을 다음과 같이 수정합니다:
```
   # -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(['gui.py'],
             pathex=[],
             binaries=[],
             datas=[('cine_settings.py', '.'), ('frame_extractor.py', '.'), ('mainclass.py', '.')],
             hiddenimports=['cv2', 'numpy', 'pycine'],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='CINE_Frame_Extractor',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
```

4. 빌드 실행: pyinstaller CINE_Frame_Extractor.spec
dist 폴더에서 생성된 CINE_Frame_Extractor.exe 파일을 확인합니다.

문제 해결

실행 파일 실행 시 오류가 발생하면 콘솔에서 실행하여 오류 메시지를 확인하세요.
안티바이러스 소프트웨어가 실행 파일을 차단할 수 있습니다. 필요한 경우 예외 처리를 해주세요.
멀티프로세싱 관련 문제 (예: 다중 창 열림)가 발생하면 최신 버전을 사용하고 있는지 확인하세요. 최신 버전에서는 멀티스레딩을 사용하여 이 문제를 해결했습니다.

변경 내역

v1.1.0: 멀티프로세싱에서 멀티스레딩으로 전환하여 윈도우 빌드 안정성 향상
v1.0.0: 최초 릴리스

라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다.
