# 머신러닝 스터디 - 오피니언 마이닝 한국어 예제

이 소스는 2015년 PyCon에서 발표된 오피니언 마이닝 한국어 예제입니다.

원 소스와 데이터는 다음 링크를 참고하였습니다.

- https://www.lucypark.kr/slides/2015-pyconkr/#36
- https://github.com/e9t/nsmc/

## 필수사항
- Python3
- Virtualenv

## 설치할 패키지
- nltk
- konlpy
- jpype1

## 디렉토리 구조
- data : 'id, 리뷰 내용, 리뷰 점수'로 구성된 CSV 파일
- cache : 위 data 디렉토리 내의 CSV 파일에서 리뷰 내용에 대해 형태소 분석한 결과에 대한 캐시

## 실행 방법
```
git clone https://github.com/jongbumi/mlstudy_opinion_mining_korean.git

cd mlstudy_opinion_mining_korean
virtualenv -p python3 venv
. venv/bin/activate

pip install nltk konlpy jpype1

python ./main.py
```
