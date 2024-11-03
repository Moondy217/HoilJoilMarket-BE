import ast
import json
# 추천
import os
import random

import numpy as np
import pandas as pd
from django.conf import settings
# 사용자
from django.contrib.auth import authenticate, login
from django.contrib.auth.hashers import make_password
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import viewsets, filters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 상품
from .models import Goods
# 주문
from .models import Orders
from .models import User
from .serializers import GoodsSerializer
from .serializers import OrdersSerializer


# 회원가입 API: 아이디, 이름, 비밀번호
@csrf_exempt
def signup_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            login_id = data.get('login_id')
            username = data.get('username')
            userpwd = data.get('userpwd')

            # 필드 확인
            if not login_id or not username or not userpwd:
                return JsonResponse({'error': 'Missing fields'}, status=400)

            # 중복 회원가입 방지
            if User.objects.filter(login_id=login_id).exists():
                return JsonResponse({'error': 'User already exists'}, status=400)

            user = User(
                login_id=login_id,
                username=username,
                userpwd=userpwd
            )
            user.save()
            return JsonResponse({'message': 'User created successfully'}, status=201)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'error': 'Invalid method'}, status=405)

# 로그인 API: 아이디, 비밀번호
@csrf_exempt
def login_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            login_id = data.get('login_id')
            userpwd = data.get('userpwd')

            if not login_id or not userpwd:
                return JsonResponse({'error': 'Missing fields'}, status=400)

            # 사용자 인증 - 아이디, 비밀번호
            user = authenticate(request, login_id=login_id, password=userpwd)

            # 사용자가 존재한다면
            if user is not None:
                login(request, user)
                return JsonResponse({'message': 'Login successful',
                                         'user': {
                                             # 로그인 성공시 프론트에 넘길 객체: 로그인하면 해당 사용자 정보로 사이트 이용
                                             'login_id': user.login_id,
                                             'userpwd':user.userpwd,
                                             'username': user.username
                                         },
                }, status=200)
            else:
                return JsonResponse({'error': 'Invalid credentials'}, status=400)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'error': 'Invalid method'}, status=405)

# 제품 관련 API: 모든 필드 사용 (등록/제거/조회/갱신)
# -- 제품 데이터에 대한 CRUD 및 검색/필터링/정렬 기능 제공
class GoodsViewSet(viewsets.ModelViewSet):
    queryset = Goods.objects.all()
    serializer_class = GoodsSerializer

    # 필터링/검색/정렬 백엔드 설정
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]

    # 필터링 필드 설정
    filterset_fields = ['category1', 'brand']

    # 검색 필드 설정
    search_fields = ['goodsName', 'goodsDesc']

    # 정렬 필드 설정
    ordering_fields = ['originalPrice', 'discountedPrice']

# 주문등록 API: 모든 필드 사용
class OrdersViewSet(viewsets.ModelViewSet):
    queryset = Orders.objects.all()
    serializer_class = OrdersSerializer


# 추천

# CSV 파일 경로 설정
CSV_DIR = os.path.join(settings.BASE_DIR, 'pm/data/')

# CSV 데이터 로드
goods = pd.read_csv(os.path.join(CSV_DIR, 'goods.csv'), encoding='utf-8-sig')
goodsKeyword = pd.read_csv(os.path.join(CSV_DIR, 'goodsKeyword.csv'), encoding='utf-8-sig')
situation = pd.read_csv(os.path.join(CSV_DIR, 'situation.csv'), encoding='utf-8-sig')
situationCategory = pd.read_csv(os.path.join(CSV_DIR, 'situationCategory.csv'), encoding='utf-8-sig')
situationKeyword = pd.read_csv(os.path.join(CSV_DIR, 'situationKeyword.csv'), encoding='utf-8-sig')

# 데이터 그룹화 및 병합
situationKeyword = situationKeyword.groupby('situationKey')['situationKeyword'].apply(list).reset_index()
situation = pd.merge(situation, situationCategory, on='situationCateKey', how='inner')
situation = pd.merge(situation, situationKeyword, on='situationKey', how='inner')

goodsKeyword = goodsKeyword.groupby('ASIN')['goodsKeyword'].apply(list).reset_index()
goods = goods.merge(goodsKeyword, on='ASIN', how='inner')

# '탕비실' 관련 데이터 필터링
pantry_situation = situation[situation['situationCategory1'] == '탕비실']
pantry_goods = goods[goods['category1'] == '식음료']
situation = situation.drop(pantry_situation.index)

# 안전한 키워드 처리 함수 (문자열을 안전하게 평가하여 변환)
def safe_literal_eval(value):
    try:
        return ' '.join(ast.literal_eval(value)) if pd.notna(value) else ''
    except (ValueError, SyntaxError):
        return str(value)

# 중복된 상품 제거 함수 (goodsKey 기준으로 중복 제거)
def remove_duplicates(goods_list):
    seen = set()
    unique_goods = []
    for item in goods_list:
        if item["goodsKey"] not in seen:
            unique_goods.append(item)
            seen.add(item["goodsKey"])
    return unique_goods

# 추천 데이터를 포맷팅하는 함수 (필요한 필드만 추출)
def format_recommendation_data(recommendation_data):
    formatted_data = []
    for headline, goods_list in recommendation_data.items():
        unique_goods_list = remove_duplicates(goods_list)
        section = headline.split(", ")[0]  # 첫 번째 헤드라인을 섹션으로 사용
        formatted_data.append({
            "section": section,
            "headline1": headline.split(", ")[0],
            "headline2": headline.split(", ")[1] if len(headline.split(", ")) > 1 else "",
            "recommendations": [
                {
                    "goodsKey": item["goodsKey"],
                    "ASIN": item["ASIN"],
                    "goodsName": item["goodsName"],
                    "originalPrice": item["originalPrice"],
                    "goodsImg": item["goodsImg"]
                }
                for item in unique_goods_list
            ]
        })
    return formatted_data

# 일반 상황에 대한 추천 함수
def situation_recommendation(situation, goods):
    random_situations = situation.sample(n=min(2, len(situation)))  # 무작위로 2개의 상황 샘플링
    recommendations = {}

    for idx, row in random_situations.iterrows():
        headline1, headline2 = row['headline1'], row['headline2']
        situation_keywords = ' '.join(row['situationKeyword'])

        # 상품 키워드 문자열로 변환
        goods_copy = goods.copy()
        goods_copy['keyword_str'] = goods_copy['goodsKeyword'].apply(safe_literal_eval)

        # TF-IDF 벡터화 및 유사도 계산
        tfidf = TfidfVectorizer()
        situation_matrix = tfidf.fit_transform([situation_keywords])
        goods_matrix = tfidf.transform(goods_copy['keyword_str'])
        similarity_scores = cosine_similarity(goods_matrix, situation_matrix).flatten()
        goods_copy['similarity'] = similarity_scores

        # 상위 50개 유사 상품 선택
        top_50_goods = goods_copy.sort_values(by='similarity', ascending=False).head(50)

        # 상위 카테고리 6개 선택
        top_categories = (
            top_50_goods.groupby(['category1', 'category2', 'category3'])
            .size()
            .reset_index(name='count')
            .sort_values(by='count', ascending=False)
            .head(6)
        )

        category_recommendations = []
        for _, category_row in top_categories.iterrows():
            category1, category2, category3 = category_row[['category1', 'category2', 'category3']]
            filtered_goods = top_50_goods[
                (top_50_goods['category1'] == category1) &
                (top_50_goods['category2'] == category2) &
                (top_50_goods['category3'] == category3)
            ].sort_values(by='similarity', ascending=False).head(15)

            sampled_goods = filtered_goods.sample(n=min(2, len(filtered_goods)), random_state=None)
            category_recommendations.append(sampled_goods)

        # 최종 추천 목록 생성
        all_recommendations = pd.concat(category_recommendations, ignore_index=True)
        if len(all_recommendations) < 12:
            remaining_goods = top_50_goods[~top_50_goods.index.isin(all_recommendations.index)]
            additional_goods = remaining_goods.head(12 - len(all_recommendations))
            all_recommendations = pd.concat([all_recommendations, additional_goods], ignore_index=True)

        final_recommendations = all_recommendations.sort_values(by='similarity', ascending=False).head(12)
        recommendations[f"{headline1}, {headline2}"] = final_recommendations.to_dict('records')

    return recommendations

# '탕비실' 상황에 대한 추천 함수
def pantry_situation_recommendation(pantry_situation, pantry_goods):
    random_situations = pantry_situation.sample(n=min(2, len(pantry_situation)))  # 무작위로 2개의 상황 샘플링
    recommendations = {}

    for idx, row in random_situations.iterrows():
        headline1, headline2 = row['headline1'], row['headline2']
        situation_keywords = ' '.join(row['situationKeyword'])

        # 상품 키워드 문자열로 변환
        pantry_goods_copy = pantry_goods.copy()
        pantry_goods_copy['keyword_str'] = pantry_goods_copy['goodsKeyword'].apply(safe_literal_eval)

        # TF-IDF 벡터화 및 유사도 계산
        tfidf = TfidfVectorizer()
        situation_matrix = tfidf.fit_transform([situation_keywords])
        goods_matrix = tfidf.transform(pantry_goods_copy['keyword_str'])
        similarity_scores = cosine_similarity(goods_matrix, situation_matrix).flatten()
        pantry_goods_copy['similarity'] = similarity_scores

        # 상위 50개 유사 상품 선택
        top_50_goods = pantry_goods_copy.sort_values(by='similarity', ascending=False).head(50)

        # 상위 카테고리 6개 선택
        top_categories = (
            top_50_goods.groupby(['category1', 'category2', 'category3'])
            .size()
            .reset_index(name='count')
            .sort_values(by='count', ascending=False)
            .head(6)
        )

        category_recommendations = []
        for _, category_row in top_categories.iterrows():
            category1, category2, category3 = category_row[['category1', 'category2', 'category3']]
            filtered_goods = top_50_goods[
                (top_50_goods['category1'] == category1) &
                (top_50_goods['category2'] == category2) &
                (top_50_goods['category3'] == category3)
            ].sort_values(by='similarity', ascending=False).head(15)

            sampled_goods = filtered_goods.sample(n=min(2, len(filtered_goods)), random_state=None)
            category_recommendations.append(sampled_goods)

        # 최종 추천 목록 생성
        all_recommendations = pd.concat(category_recommendations, ignore_index=True)
        if len(all_recommendations) < 12:
            remaining_goods = top_50_goods[~top_50_goods.index.isin(all_recommendations.index)]
            additional_goods = remaining_goods.head(12 - len(all_recommendations))
            all_recommendations = pd.concat([all_recommendations, additional_goods], ignore_index=True)

        final_recommendations = all_recommendations.sort_values(by='similarity', ascending=False).head(12)
        recommendations[f"{headline1}, {headline2}"] = final_recommendations.to_dict('records')

    return recommendations

# 추천 API 함수 (상황별 추천 데이터 반환)
def recommend(request):
    # 일반 상황과 '탕비실' 상황에 대해 각각 추천 수행
    situation_data = situation_recommendation(situation, goods)
    pantry_data = pantry_situation_recommendation(pantry_situation, pantry_goods)

    # 추천 결과 형식화
    formatted_situation_data = format_recommendation_data(situation_data)
    formatted_pantry_data = format_recommendation_data(pantry_data)

    # 결과를 JSON 형태로 반환
    combined_result = {
        "situation_recommendations": formatted_situation_data,
        "pantry_recommendations": formatted_pantry_data
    }

    return JsonResponse(combined_result, safe=False)