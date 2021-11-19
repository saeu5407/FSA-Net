# FSA-Net

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fsaeu5407&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

**원본 FSA-Net Link : [FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation from a Single Image](https://github.com/shamangary/FSA-Net)**

**현재 진행중인 프로젝트에서 FSA-Net을 사용하고 있습니다. 이왕 FSA-Net에 대해 알아본김에 `demo/run_demo_FSANET_ssd.py`에 대해 주석을 달아놓은 파일을 공개해두고자 생성한 REPO입니다.**

**시간이 될 때 블로그에 논문 리뷰글을 올리겠습니다. [BLOG | dyddl1993](https://dyddl1993.tistory.com)**

---

### **Facial Detect Model**

여기서는 `DEMO`의 3가지 버젼에 대해서 간략하게 정리하겠습니다.<br>
FSA-Net 앞단의 `Facial Detect Model`은 총 세 가지 옵션이 있습니다. 각각을 편하게 `Haar`, `MTCNN`, `SSD`라고 하겠습니다.

`Haar`의 경우 가장 대중적이고 기본적인 모델입니다. 가볍다는 장점이 있지만 성능이 뛰어나지는 않습니다.

`MTCNN`의 경우 멀티-태스크 러닝 모델로 좋은 성능과 더불어 5개의 랜드마크도 확인할 수 있습니다. 
다만 많이 무겁다는 단점과, 랜드마크 방식의 문제인지는 모르겠지만 마스크를 쓰고 있을 때 등에서 성능이 확 떨어지는 단점을 가지고 있습니다. 
MTCNN도 논문을 정리해 본 적이 있어서 향후에 블로그에 간단하게 리뷰를 올려보도록 하겠씁니다.

`SSD` 방식은 pre-trained된 모델을 가져와 사용하는 방식으로, FSA-Net에서는 resnet을 사용하고 있는 것으로 보입니다. 
성능이 제일 좋고 빠르지만, 그만큼 자원을 많이 필요로 합니다. 

제가 작업중인 프로젝트에서는 `HOG` 방식 등 다양한 방법들을 비교해보고, 최종적으로는 `SSD`를 `Facial Detect Model`로 사용하기로 결정했습니다.

참고할 만한 블로그는 다음과 같습니다. [Face detection model 성능 비교(WIDERFace) | seongkyun](https://seongkyun.github.io/study/2019/03/25/face_detection/)