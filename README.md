# 문장 내 개체간 관계 추출

- train.csv: 총 32470개
- test_data.csv: 총 7765개

- Data 
![16bc4f53-f355-4b9d-968f-657bb5d9b5e5](https://user-images.githubusercontent.com/62659407/162699051-871a1d3a-f249-4d90-a9da-83334e0af681.png)  

- Class
![3f0beeb4-199e-412c-9a41-a4423582b506](https://user-images.githubusercontent.com/62659407/162699171-4006c2f6-739f-493a-9bef-f06d23f2a11e.png)

1. EDA   
    - label            
![Untitled](https://user-images.githubusercontent.com/62659407/162708122-a08c6102-f2ae-4d86-a130-16100d9c1e92.png)

    - 문장 분포 확인           
![output1](https://user-images.githubusercontent.com/62659407/162708287-d33c38f4-53aa-4a0b-904e-3e914f36e108.png)

    - label별 문장 분포 확인         
![output5](https://user-images.githubusercontent.com/62659407/162708316-a2687b19-e2d4-443f-8015-b8dd0b08fd31.png)

    - 학습 데이터의 Subject, Object 단어의 type 분포 확인
![output6](https://user-images.githubusercontent.com/62659407/162708440-acef8e44-0ac3-4ba2-9793-1104ec93b4a9.png)

    - 테스트 데이터의 Subject, Object 단어의 type 분포 확인          
![output7](https://user-images.githubusercontent.com/62659407/162708383-a995502e-7dd5-402d-a513-9986792a043b.png)

    - 학습 데이터 : id-subj_info-obj_info-sen-label-label_num 순서로 구조 재구성            
![개선된 df JPG](https://user-images.githubusercontent.com/62659407/162708483-36c3b3d7-3c4f-441e-bcd0-b1eed567aa63.jpg)

    - 중복 데이터 : 84개의 중복 데이터 확인 후 제거
    - 오태깅 데이터 제거 : 5개의 오태깅 데이터 확인 후 제거
    - 데이터 교정 : 학습 데이터 중 `subj_type`, `obj_type`, `label` 이 잘못된 데이터를 교정 → 오히려 성능이 떨어져 원래의 학습 데이터를 사용
    - Easy Data Augmentation : KoEDA 라이브러리를 사용하여 Random Insertion, Random Deletion, Random Swap, Synonym Replacement 적용 → 성능 개선 효과 없었음
        - 논문 “[1901.11196.pdf (arxiv.org)](https://arxiv.org/pdf/1901.11196.pdf)” 참고

2. Preprocess
    - Typed Entity marker(punct)
        - 논문 “An Improved Baseline for Sentence-level Relation Extraction” 참고
        ![1 (11)](https://user-images.githubusercontent.com/62659407/162708505-4830f879-26e8-4850-9639-41a729ac3665.png)
        
        원본 : 〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.
        
        → Typed Entity marker(punct) : 〈Something〉는 # ^ [PER] ^ 조지 해리슨 # 이 쓰고 @ * [PER] * 비틀즈 @ 가 1969년 앨범 《Abbey Road》에 담은 노래다. 
        
        ⇒  [CLS] 〈Something〉는 # ^ [PER] ^ 조지 해리슨 # 이 쓰고 @ * [PER] * 비틀즈 @ 가 1969년 앨범 《Abbey Road》에 담은 노래다. [SEP]
        
    
    - Typed Entity marker(punct) + Query
        - 논문 “BERT: Pre-training of Deep Bidirectional Transformers for
        Language Understanding” 참고
   ![1 (12)](https://user-images.githubusercontent.com/62659407/162708551-bc264b6c-1637-48b3-b8c6-434782f87f59.png)
        
        ⇒ 기존 BERT의 Pretain 방식과 유사한 input으로 만들어줌
        
        원본 : 〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.
        
        → Typed Entity marker(punct) : 〈Something〉는 # ^ [PER] ^ 조지 해리슨 # 이 쓰고 @ * [PER] * 비틀즈 @ 가 1969년 앨범 《Abbey Road》에 담은 노래다.
        
        → Query : @ * [PER] * 비틀즈 @ 와 # ^ [PER] ^ 조지 해리슨 # 의 관계 
        
        ⇒ [CLS] @ * [PER] * 비틀즈 @ 와 # ^ [PER] ^ 조지 해리슨 # 의 관계 [SEP] 〈Something〉는 # ^ [PER] ^ 조지 해리슨 # 이 쓰고 @ * [PER] * 비틀즈 @ 가 1969년 앨범 《Abbey Road》에 담은 노래다. [SEP]
        
    - Standard with Entity Location Token
        - 논문 “엔티티 위치 정보를 활용한 한국어 관계 추출 모델 비교 및 분석” 참고
![1 (13)](https://user-images.githubusercontent.com/62659407/162708582-0b8576e2-0101-40b6-8c95-a79fb6c8b32e.png)
        
        원본 : 〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.
        
        → Standard with Entity Location Token : 〈Something〉는 [OBJ] 조지 해리슨 [/OBJ] 이 쓰고 [SUB] 비틀즈 [/SUB] 가 1969년 앨범 《Abbey Road》에 담은 노래다.
        
        ⇒ [CLS] 〈Something〉는 [OBJ] 조지 해리슨 [/OBJ] 이 쓰고 [SUB] 비틀즈 [/SUB] 가 1969년 앨범 《Abbey Road》에 담은 노래다. [SEP]
        
    - Backtranslation : Selenium을 활용한 크롤링을 통해 한국어 → 영어 → 한국어 번역
        
        
        원본 : 〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.
        
        → "Something" is a song written by George Harrison and included by the Beatles on their 1969 album Abbey Road.
        
        ⇒ "Something"은 조지 해리슨이 작곡하고 비틀즈가 1969년 앨범 Abbey Road에 포함시킨 노래입니다.
        
    
    - 성능 비교(micro f1)
        
        (AutoModelForSequenceClassification.from_pretrained("klue/roberta-large"))
        
        - Typed Entity marker(punct) : 71%
        - **Typed Entity marker(punct) + Query : 73%**
        - Standard with Entity Location Token : 70%
        - Backtranslation : 72% → 생성 문장을 살펴보면 저품질 문장이 많음

3. Model
    - Pretrained Model
        - klue/bert-base
        - **klue/roberta-large**

    - Additional Layer
        - AutoModelForSequenceClassification
![fc](https://user-images.githubusercontent.com/62659407/162708635-e991cd6a-abf4-422a-b38e-8db222b33888.png)

        - FC
![fc1](https://user-images.githubusercontent.com/62659407/162708661-6e68cfa8-b96a-4577-bca1-c1196ad28f21.png)

        - BiLSTM
![bilstm](https://user-images.githubusercontent.com/62659407/162708678-68339a19-00ba-4fd3-9626-4b7fa7881fa0.png)

        - BiGRU
![bigru](https://user-images.githubusercontent.com/62659407/162708706-83b17d1b-d9cb-426a-84db-8832db12c1d6.png)

        - BiGRU + Multi-Head-Attention + BiGRU
            - 논문 “UO UP V2 at HAHA 2019: BiGRU Neural
            Network Informed with Linguistic Features for
            Humor Recognition” 참고
![mha](https://user-images.githubusercontent.com/62659407/162708738-50e3250c-4f6f-4dbd-9be6-085e2ff0da3f.png)

- 성능 비교(micro f1)

    (Typed Entity marker(punct) + Query)

    - AutoModelForSequenceClassification : 74.9%
    - FC : 74.3%
    - **BiLSTM : 75.6%**
    - BiGRU : 75.1%
    - BiGRU + Multi-Head-Attention + BiGRU : 74.4%

4. Hyper Parameter
    - Learning Rate
        - **3e - 5**
        - **warmup_ratio : 0.1**
        - **decay to 0**
        - **weight decay : 0.01** → overfitting 방지

        → 논문 “An Improved Baseline for Sentence-level Relation Extraction” 참고
![1 (14)](https://user-images.githubusercontent.com/62659407/162708786-3b116f70-a1de-4593-849b-073fe4c8dff9.png)

    - Batch Size
        - **64**
        - 32

        → **Out Of Memory가 일어나지 않는 선에서 Batch Size는 클수록 성능이 좋았음**


    - max_len : 입력 Sentence의 최대 길이
        - **160**
        - 256

        → **성능은 비슷했지만, 256은 batch size를 64로 했을 때, Out Of Memory가 발생해 160을 사용**

    - Epoch
        - **5**
        - 10

        → **똑같은 조건에서 Epoch가 10일 때, 성능이 더 떨어졌고, Overfitting이 발생했다고 판단**

    - Loss Function
        - **Cross Entropy** : Transformer의 Default
        - Focal Loss : Class Imabalance를 개선하지만 CE와 성능 차이가 없었음
        - **Label Smoothing : 0.1** → Class Imbalance 개선

    - Optimizer
        - **AdamW** : Transformer의 Default

5. Train
    - StratifiedKFold : 성능 개선 효과 없었음

6. SOTA 모델
    - Preprocess : Typed Entity marker(punct) + Query
    
    - Model
        - Pretrained Model : “klue/roberta-large”
        - Additional Layer : BiLSTM
    
    - Hyper Parameter
        - Learning Rate
            - 3e - 5
            - warmup_ratio : 0.1
            - decay to 0
            - weight decay : 0.01
        
        - Batch Size : 64
        - max_len : 160
        - Epoch : 5
        - Loss Function
            - Cross Entropy
            - Label Smoothing : 0.1
        
        - Optimizer : AdamW

7. Ensemble
    - **Soft Voting**
        - 리더보드 상위 4개 모델 : **76.7338% (SOTA + 1%)**
        - 5개의 모델 : 75.5%
            - AutoModelForSequenceClassification : 74.9%
            - FC : 74.3%
            - BiLSTM : 75.6%
            - BiGRU : 75.1%
            - BiGRU + Multi-Head-Attention + BiGRU : 74.4%
        - 상위 3개의 모델 : 75.7%
            - AutoModelForSequenceClassification : 74.9%
            - BiLSTM : 75.6%
            - BiGRU : 75.1%
            
8. 최종 결과
    - Public(11팀 中 2등)
     ![1 (15)](https://user-images.githubusercontent.com/62659407/162708823-5ea5f4b8-5d7d-4021-9d6a-4300b851c001.png)
   
    - Private(11팀 中 3등)
      ![1 (16)](https://user-images.githubusercontent.com/62659407/162708852-9bfdb2bf-649d-427a-98d2-d471a7861704.png)
