"""
Nemotron-Personas-Korea EDA + 최애 주민 뽑기
한양대 ERICA 캡스톤 (롯데이노베이트) 숙제용
"""

# ===== 0. 설치 (최초 1회) =====
# pip install datasets pandas matplotlib

from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

# ===== 1. 한글 폰트 설정 =====
#   !apt-get -qq install fonts-nanum > /dev/null
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# ===== 2. 데이터 로드 =====
ds = load_dataset("nvidia/Nemotron-Personas-Korea")
df = ds["train"].to_pandas()

print("shape:", df.shape)
print("columns:", df.columns.tolist())
print(df.head(3))

# ===== 3. 기본 EDA =====
print("\n[결측치]")
print(df.isnull().sum())

print("\n[성별]")
print(df["sex"].value_counts())

print("\n[나이 통계]")
print(df["age"].describe())

print("\n[시도별 상위 10]")
print(df["province"].value_counts().head(10))

print("\n[학력 분포]")
print(df["education_level"].value_counts())

print("\n[혼인 상태]")
print(df["marital_status"].value_counts())

print("\n[직업 상위 20]")
print(df["occupation"].value_counts().head(20))

# ===== 4. 시각화 (4분할) =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

df["age"].hist(bins=30, ax=axes[0, 0])
axes[0, 0].set_title("연령 분포")

df["province"].value_counts().head(10).plot(kind="barh", ax=axes[0, 1])
axes[0, 1].set_title("시도별 인원 (상위 10)")
axes[0, 1].invert_yaxis()

df["education_level"].value_counts().plot(kind="bar", ax=axes[1, 0])
axes[1, 0].set_title("학력 분포")
axes[1, 0].tick_params(axis="x", rotation=45)

df["marital_status"].value_counts().plot(kind="pie", ax=axes[1, 1], autopct="%1.1f%%")
axes[1, 1].set_title("혼인 상태")
axes[1, 1].set_ylabel("")

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=150)
plt.show()

# ===== 5. 프로젝트 연계 미니 분석 =====
# 롯데이노베이트 사업장(서울 금천구) 근방 페르소나 규모 확인
# -> 실제 시뮬레이션에서 특정 행정동 샘플링할 때 이런 식으로 필터링하게 됨 (모듈①)
geumcheon = df[df["district"].str.contains("금천구", na=False)]
print(f"\n금천구 거주 페르소나 수: {len(geumcheon)}")
print(geumcheon["occupation"].value_counts().head(10))

# ===== 6. 최애 주민 뽑기 =====

def print_persona(row: pd.Series) -> None:
    fields = [
        "uuid", "sex", "age", "occupation", "district", "province",
        "education_level", "marital_status", "family_type",
        "persona", "professional_persona",
        "hobbies_and_interests", "skills_and_expertise",
        "career_goals_and_ambitions",
    ]
    for f in fields:
        print(f"[{f}]\n{row[f]}\n")

# 방법 A: 완전 랜덤
random_persona = df.sample(1, random_state=42).iloc[0]
print("===== 랜덤 픽 =====")
print_persona(random_persona)

# 방법 B: 관심사 기반 필터링 (본인 관심사에 맞게 키워드 수정해서 사용)
# 예시: 백엔드/인프라/네트워크 쪽 직업·스킬을 가진 페르소나
keyword_mask = (
    df["occupation"].str.contains("네트워크|정보통신|보안|데이터", na=False)
    | df["skills_and_expertise"].str.contains("데이터베이스|서버|네트워크|클라우드", na=False)
)
age_mask = df["age"].between(25, 40)
tech_personas_young = df[keyword_mask & age_mask]
print(f"\n키워드 조건에 맞는 페르소나 수: {len(tech_personas_young)}")
 
if len(tech_personas_young) > 0:
    my_favorite = tech_personas_young.sample(1, random_state=1).iloc[0]
    print("\n===== 조건부 픽 (관심사 필터링) =====")
    print_persona(my_favorite)
