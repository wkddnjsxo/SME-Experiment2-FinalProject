# 스마트 모빌리티 공학실험 Final Project

<div align="right">

**학번:** 12223649
**이름:** 장원태

</div>

# 모티베이션 & 인트로

본 프로젝트의 목표는 18개 기지국에서 측정된 RTT 기반 거리값을 이용하여 사용자 단말의 2차원 위치를 추정하는 것이다. 입력으로는 각 사용자에 대한 측정 거리 벡터와 기지국 좌표가 주어지며, 최종 출력은 사용자 위치의 예측값이다. 처음에는 거리값과 기지국 좌표가 주어지므로 삼변측량 또는 일반적인 least squares 기반 위치추정만으로 충분할 것이라고 예상했다. 그러나 중간 실험 과정에서 실제 RTT 측정값은 모든 기지국에서 균일하게 신뢰할 수 있는 값이 아니며, 특정 기지국이나 특정 위치 영역에서 오차가 크게 발생한다는 점을 확인하였다.

초기 실험에서는 raw RTT 거리값을 그대로 사용한 위치추정, 거리 보정 기반 위치추정, 고정 가중치를 사용한 weighted least squares, 그리고 GNN을 이용한 anchor reliability 학습을 단계적으로 비교하였다. 이 과정에서 단순히 모든 기지국을 동일하게 신뢰하는 방식은 큰 오차가 발생할 수 있고, 반대로 기지국별 측정 신뢰도를 데이터에서 학습하면 위치추정 성능이 크게 개선될 수 있음을 확인하였다. 특히 이전 탐색 실험에서 고정 가중치 기반 WLS보다 GNN이 학습한 anchor weight를 WLS에 반영한 구조가 더 낮은 위치 오차를 보였다.

이 관찰을 바탕으로 최종 알고리즘은 단순 삼변측량이 아니라, 기지국별 측정 신뢰도를 GNN으로 학습하고 이를 weighted least squares 위치 계산에 반영하는 hybrid localization model로 설계하였다. 즉, 거리와 위치 사이의 기하학적 관계는 WLS 계층이 담당하고, 어떤 기지국의 측정값을 더 신뢰할지는 신경망이 학습한다. 또한 WLS 결과만으로 설명되지 않는 잔차 오차를 보정하기 위해 residual correction을 추가하였다.

최종 제출 모델은 hidden test 환경을 고려하여, 별도의 중간 CSV 파일이나 validation 정답 정보에 의존하지 않도록 구성하였다. 학습에는 제공된 DH_FR1.mat의 정답 위치를 사용하지만, main.py에서 추론할 때는 측정 거리값과 기지국 좌표, 그리고 train.py로 저장한 model.pt만 사용한다. 따라서 채점 데이터에서 사용자 수가 달라져도 입력 데이터의 사용자 수를 동적으로 읽어 예측 위치 배열을 반환할 수 있다.

# 알고리즘 설명

각 사용자에 대해 18개 기지국의 측정 거리값을 $d = [d_1, d_2, \cdots, d_{18}]$로 정의하고, $i$번째 기지국의 좌표를 $s_i = (x_i, y_i)$로 정의한다. 예측해야 하는 사용자 위치는 $p = (x, y)$이다. 모델의 최종 목표는 측정 거리값과 기지국 좌표를 입력받아 예측 위치 $\hat{p}$를 출력하는 것이다.

본 알고리즘은 특정 논문 하나를 그대로 재구현한 것이 아니라, 여러 기존 연구의 핵심 아이디어를 RTT localization 문제에 맞게 결합한 구조이다. Beck, Stoica, and Li의 source localization 연구에서는 range measurement를 이용한 least squares 기반 위치추정 문제를 다루며, 본 연구는 여기서 거리식 기반 위치추정과 LS/WLS 문제 설정을 참고하였다. 다만 해당 연구는 주로 수식적 LS 해법에 초점을 두는 반면, 본 연구는 각 기지국의 가중치를 고정하지 않고 GNN이 데이터 기반으로 학습하도록 설계하였다.

Gilmer et al.의 Message Passing Neural Network 연구와 Kipf and Welling의 Graph Convolutional Network 연구에서는 graph node들이 서로의 정보를 집계하여 node 또는 graph representation을 학습하는 개념을 제시한다. 본 연구는 이 graph message passing 개념을 분자 그래프나 citation graph가 아니라 18개 RTT 기지국 그래프에 적용하였다. 따라서 본 연구에서 node는 기지국이고, node feature는 기지국 좌표와 측정 거리 관련 feature이며, node output은 classification label이 아니라 WLS에 사용되는 anchor reliability score이다.

He et al.의 residual learning 연구에서는 기준 mapping을 그대로 학습하기보다 기준값에 대한 잔차를 학습하는 방식이 효과적임을 보였다. 본 연구는 ResNet 구조를 이미지 인식에 그대로 적용한 것이 아니라, WLS가 계산한 1차 위치 $p_{\mathrm{wls}}$를 기준값으로 두고, 신경망이 잔차 보정량 $\Delta p$를 학습하도록 residual learning 개념을 위치추정 문제에 응용하였다. 정리하면, 참고 논문에서 가져온 부분은 range-based WLS 문제 설정, graph message passing, residual learning 개념이고, 본 연구에서 직접 설계한 부분은 이 세 가지를 RTT 기지국 신뢰도 학습 문제로 연결하여 anchor별 거리 보정, GNN reliability weight, pairwise WLS, residual correction을 하나의 end-to-end 모델로 구성한 점이다.

첫 번째 단계는 기지국별 거리 보정이다. 실제 RTT 기반 거리값은 기지국마다 편향과 scale 차이를 가질 수 있으므로, 모델은 각 기지국에 대해 학습 가능한 거리 보정 파라미터를 둔다. 보정된 거리는 다음과 같이 표현할 수 있다.

$$
d'_i = \mathrm{softplus}(a_i)d_i + b_i
$$

여기서 $a_i$와 $b_i$는 $i$번째 기지국에 대한 학습 파라미터이다. $\mathrm{softplus}(\cdot)$를 사용하는 이유는 거리 scale이 음수가 되는 것을 방지하기 위해서이다. 이 단계는 기존의 고정 보정식을 사용하는 대신, 위치 예측 loss를 통해 각 기지국의 거리 보정 특성을 end-to-end로 학습하게 만든다. 이전 중간 실험에서는 RANSAC, Isotonic Regression, Quantile Regression을 사용하여 거리 보정과 불확실성 추정을 별도 CSV 파일로 만들었지만, 최종 제출 모델에서는 hidden test에서 사용할 수 없는 중간 파일 의존성을 제거하였다. 대신 raw $d_i$를 입력받아 모델 내부에서 각 anchor별 scale과 bias를 학습하도록 바꾸었다.

두 번째 단계는 Anchor Reliability GNN이다. 본 알고리즘은 18개 기지국을 각각 하나의 노드로 보고, 각 노드에 보정 거리, 기지국 좌표, 거리의 상대적 크기, 측정값의 공간적 일관성에 대한 feature를 부여한다. 이후 message passing을 통해 각 기지국을 독립적으로 판단하지 않고, 다른 기지국들과의 관계 속에서 해당 기지국 측정값의 신뢰도를 추정한다. GNN의 출력은 각 기지국에 대한 reliability score이며, 이를 양수 가중치로 변환하여 WLS에 사용한다.

$$
w_i = \mathrm{softplus}(g_i) + \epsilon
$$

여기서 $g_i$는 GNN이 출력한 $i$번째 기지국의 raw reliability score이고, $w_i$는 weighted least squares에 사용되는 기지국 신뢰도이다. $\epsilon$은 수치적으로 0에 가까운 가중치로 인한 불안정성을 막기 위한 작은 양수이다. 이 부분이 본 연구의 핵심 차별점이다. 기존 WLS는 사람이 사전에 정한 가중치나 모든 anchor에 동일한 가중치를 사용하는 경우가 많지만, 본 연구에서는 각 사용자의 측정 거리 패턴과 anchor 간 일관성을 보고 GNN이 동적으로 anchor reliability를 결정한다.

세 번째 단계는 GNN이 예측한 신뢰도 가중치를 이용한 pairwise weighted least squares 위치 계산이다. 거리 기반 위치추정의 기본 목적함수는 다음과 같이 표현할 수 있다.

$$
\min_p \sum_i w_i \left(\|p - s_i\| - d'_i\right)^2
$$

이 식은 $p$에 대해 비선형이므로 직접 반복 최적화를 수행할 수도 있지만, 본 알고리즘에서는 계산 속도와 안정성을 위해 거리식의 차이를 이용한 선형화된 WLS를 사용한다. 각 기지국에 대해 $\|p - s_i\|^2 = d_i'^2$ 형태의 식을 세우고, 서로 다른 두 기지국의 식을 빼면 $p$에 대한 선형식이 만들어진다. 여러 anchor pair에 대해 이러한 선형식을 쌓으면 다음과 같은 형태가 된다.

$$
A p = b
$$

이때 pairwise WLS 해는 다음과 같이 계산된다.

$$
p_{\mathrm{wls}} = (A^T W A + \lambda I)^{-1} A^T W b
$$

여기서 $W$는 두 anchor의 reliability weight를 조합하여 만든 pair weight이고, $\lambda I$는 행렬 역계산의 수치적 안정성을 높이기 위한 regularization 항이다. 이 구조를 사용하면 모델이 완전히 블랙박스로 좌표를 회귀하는 것이 아니라, 거리와 좌표 사이의 기하학적 관계를 명시적으로 반영할 수 있다.

네 번째 단계는 residual correction이다. WLS는 기하학적으로 해석 가능한 위치를 제공하지만, 실제 RTT 오차에는 비선형 편향과 이상치 영향이 남아 있을 수 있다. 따라서 모델은 WLS 위치 $p_{\mathrm{wls}}$에 대해 작은 보정량 $\Delta p$를 추가로 예측한다.

$$
p_{\mathrm{final}} = p_{\mathrm{wls}} + \Delta p
$$

$\Delta p$는 GNN에서 얻은 전체 anchor feature와 WLS 결과를 기반으로 예측된다. 이 구조의 목적은 WLS가 담당하는 큰 위치 구조는 유지하면서, 데이터에서 반복적으로 나타나는 잔차 패턴만 신경망이 보정하도록 만드는 것이다. 최종적으로 main.py는 모든 사용자에 대해 $p_{\mathrm{final}}$을 계산하고, 이를 모아 shape이 $(2, \mathrm{num\_user})$인 $\hat{p}$를 반환한다.

학습 loss는 최종 위치 예측 오차를 중심으로 구성하되, WLS 위치 자체도 너무 나빠지지 않도록 보조 loss를 포함한다. 또한 residual correction이 지나치게 커져서 WLS의 기하학적 의미를 무너뜨리지 않도록 보정량에 대한 regularization을 적용한다. 최종적으로 모델은 train.py에서 학습되고, 학습된 파라미터와 정규화 통계량은 model.pt로 저장된다. main.py는 이 model.pt를 불러와 hidden test의 $d_{hat}$과 기지국 좌표만으로 위치를 추론한다.

# Agent AI 활용 방안

본 프로젝트에서는 Agent AI를 알고리즘의 최종 판단자가 아니라, 실험 설계 보조, 코드 구조화, 디버깅, 보고서 정리 도구로 활용하였다. 최종적인 알고리즘 선택과 결과 해석은 validation 결과와 hidden test 일반화 가능성을 기준으로 직접 판단하였다.

| 구분 | 본인의 역할 | Agent AI의 역할 |
|---|---|---|
| 문제 이해 | RTT 기반 위치추정 문제와 제출 형식을 확인하고, hidden test에서 사용 가능한 입력을 구분하였다. | README 조건을 정리하고, main.py와 train.py가 만족해야 하는 입출력 형식을 설명하였다. |
| 알고리즘 설계 | 단순 WLS, 거리 보정, GNN anchor weight, residual correction 중 최종적으로 사용할 구조를 선택하였다. | 각 방법의 장단점과 보고서에서 설명 가능한 알고리즘 구조를 제안하였다. |
| 구현 | 컨테이너 환경에서 코드를 실행하고, 학습 로그와 오류 메시지를 확인하였다. | PyTorch 기반 학습 코드 구조, 모델 저장 및 로드 방식, Docker 환경 설정을 보조하였다. |
| 디버깅 | 실제 .mat 파일의 변수명을 확인하고, 학습 결과가 정상적으로 저장되는지 검증하였다. | p_bs와 BS_positions 변수명 차이, CPU 실행 여부, early stopping 동작 원인을 설명하였다. |
| 결과 해석 | validation MAE, median, RMSE, P90을 기준으로 제출 후보 모델을 선택하였다. | 단일 validation split 결과와 이전 OOF 실험 결과를 공정하게 구분하여 해석하는 방향을 제안하였다. |
| 보고서 작성 | 최종 알고리즘의 선택 이유와 실험 결과를 본인의 관점에서 정리하였다. | 문장 구조, 수식 설명, markdown 표 구성 방식을 보조하였다. |

Agent AI를 통해 단순히 코드를 생성한 것이 아니라, 실험 중 발생한 문제를 빠르게 해석하고 대안을 비교하는 방식으로 활용하였다. 예를 들어 CNN 기반 heatmap localization도 후보로 검토했지만, 학습 시간이 길고 빠른 검증 결과가 기대보다 좋지 않았기 때문에 최종 제출 모델에서는 제외하였다. 반대로 GNN-WLS 구조는 위치추정 문제의 기하학적 성질을 유지하면서도 anchor별 신뢰도를 학습할 수 있어, 데이터 수가 많지 않은 상황에서 더 적합하다고 판단하였다.

# 결과 도출 & 디스커션

최종 제출 모델은 제공된 700명 데이터에서 train/validation split을 사용하여 학습하였다. 학습은 최대 600 epoch로 설정하였고, validation MAE가 가장 낮았던 epoch의 모델을 model.pt로 저장하였다. 마지막 실행에서는 370 epoch 부근에서 가장 낮은 validation MAE를 기록하였으며, 최종 저장 모델의 결과는 다음과 같다.

| 모델 | 평가 방식 | MAE (m) | Median (m) | RMSE (m) | P90 (m) |
|---|---|---:|---:|---:|---:|
| Final Anchor Reliability GNN-WLS | validation split | 2.1515 | 1.4260 | 3.0933 | 4.6332 |

이 결과는 train.py 내부의 validation split에서 얻은 값이므로, hidden test 성능과 완전히 동일하다고 볼 수는 없다. 그러나 학습에 사용하지 않은 validation set에서 평균 오차가 약 2.15 m이고, 90% 오차가 약 4.63 m라는 점은 모델이 단순히 train set을 외운 것만은 아니라는 근거가 된다. 또한 main.py를 실행했을 때 제공 데이터 700명에 대해 shape이 (2, 700)인 예측 배열이 정상적으로 반환되는 것을 확인하였다.

중간 실험 과정에서 사용한 탐색 결과는 다음과 같다. 단, 아래 결과들은 최종 제출 모델과 완전히 같은 입력 조건이나 평가 방식이 아니므로, 직접적인 우열 비교가 아니라 알고리즘 선택을 위한 실험적 근거로 해석하였다.

중간 실험에서 비교한 모델명은 다음과 같은 의미를 가진다. Fixed uncertainty-prior WLS는 신경망 학습 없이 사전에 계산한 거리 불확실성 기반 가중치만 사용하여 WLS를 수행한 방법이다. GNN learned-weight WLS only는 GNN이 각 기지국의 신뢰도 가중치를 학습하고, 이 가중치만 WLS에 반영한 모델이다. 이 경우 최종 위치는 WLS 결과 $p_{\mathrm{wls}}$를 그대로 사용하며 residual correction은 적용하지 않는다. GNN WLS + residual correction은 GNN이 학습한 anchor weight로 WLS 위치를 계산한 뒤, 추가적인 잔차 보정량 $\Delta p$를 더해 최종 위치를 얻는 구조이다. 마지막으로 Final submission model은 위 구조를 제출 환경에 맞게 재구성한 모델로, hidden test에서 사용할 수 없는 중간 CSV 파일 없이 DH_FR1.mat의 $d_{hat}$과 기지국 좌표만으로 동작하도록 만든 최종 모델이다.

| 실험 단계 | 평가 방식 | 주요 결과 | 해석 |
|---|---|---:|---|
| Fixed uncertainty-prior WLS | 5-Fold OOF | MAE 9.5757 m | 고정 가중치만으로는 anchor별 오차 차이를 충분히 반영하지 못하였다. |
| GNN learned-weight WLS only | 5-Fold OOF | MAE 3.4483 m | GNN이 anchor 신뢰도를 학습하면서 WLS 위치 오차가 크게 감소하였다. |
| GNN WLS + residual correction | 5-Fold OOF | MAE 2.9426 m | WLS 이후 남는 잔차를 보정하여 추가적인 성능 개선이 있었다. |
| Final submission model | validation split | MAE 2.1515 m | 제출 형식에 맞춰 .mat 단독 입력으로 재구성한 최종 모델이다. |

비교의 공정성 측면에서, 딥러닝 모델과 단순 삼변측량의 수치만 직접 비교하는 것은 적절하지 않다고 판단하였다. 단순 삼변측량은 학습 데이터와 정답 위치를 사용하지 않는 반면, 본 모델은 제공 데이터의 정답 위치를 이용해 anchor reliability와 residual correction을 학습하기 때문이다. 따라서 본 보고서에서는 단순 WLS 대비 성능 향상을 주장하기보다는, “거리 기반 위치추정의 기하학적 구조를 유지하면서, 측정 신뢰도만 데이터 기반으로 학습하는 hybrid 방식이 타당하다”는 점을 중심으로 해석하였다.

본 알고리즘의 장점은 첫째, 거리와 좌표 사이의 물리적 관계를 WLS 계층으로 유지한다는 점이다. 완전한 direct coordinate regression은 데이터가 적을 때 overfitting될 위험이 있지만, WLS 구조를 사용하면 거리 기반 위치추정 문제의 기본 형태를 보존할 수 있다. 둘째, 모든 기지국을 동일하게 신뢰하지 않고 GNN이 anchor별 reliability를 학습한다. 이는 특정 기지국의 측정값이 불안정하거나 특정 위치에서 오차가 커지는 상황에 대응할 수 있다. 셋째, residual correction을 통해 WLS만으로 설명되지 않는 비선형 오차를 보정할 수 있다. 넷째, 최종 main.py는 hidden test에서 사용할 수 없는 중간 CSV 파일에 의존하지 않고, d_hat과 기지국 좌표, model.pt만으로 추론할 수 있다.

반면 한계도 존재한다. 첫째, 최종 제출 모델의 성능 평가는 단일 validation split에 기반하므로 split에 따라 결과가 달라질 수 있다. 둘째, hidden test의 위치 분포나 RTT 오차 분포가 제공 데이터와 크게 다르면 학습된 reliability weight와 residual correction이 충분히 일반화되지 않을 수 있다. 셋째, 매우 큰 이상치가 포함된 거리값에 대해서는 WLS 위치와 residual correction이 모두 영향을 받을 수 있다. 넷째, model.pt 파일이 필요하므로 main.py 단독으로는 학습된 모델 성능을 재현할 수 없고, train.py를 통해 같은 학습 과정을 재현할 수 있어야 한다.

향후 개선 방향으로는 K-Fold ensemble을 적용하여 split 의존성을 줄이는 방법이 있다. 예를 들어 5개의 fold 모델을 각각 학습한 뒤 hidden test에서 평균 예측을 사용하면 단일 split 모델보다 안정적인 결과를 기대할 수 있다. 또한 q10, q50, q90 형태의 거리 불확실성 추정을 최종 .mat 단독 구조 안에 통합하면, GNN이 단순한 거리값뿐 아니라 측정 불확실성까지 함께 고려할 수 있다. 마지막으로 Huber weight나 RANSAC 기반 pair filtering을 WLS 계층에 추가하면 극단적인 RTT 이상치에 대한 강건성을 높일 수 있다.

결론적으로 본 프로젝트의 최종 알고리즘은 단순 삼변측량이나 완전한 black-box regression 중 하나를 선택한 것이 아니라, 두 접근의 장점을 결합한 hybrid 구조이다. 기하학적 위치 계산은 WLS로 수행하고, 기지국별 신뢰도와 잔차 보정은 GNN이 학습하도록 분리함으로써, 제공된 데이터 수가 제한적인 상황에서도 비교적 안정적인 위치추정 성능을 얻을 수 있었다.

# Reference

본 연구는 특정 논문 하나를 그대로 재구현한 것이 아니라, 거리 기반 위치추정의 WLS 구조, graph message passing, residual correction 개념을 RTT localization 문제에 맞게 결합한 것이다. 아래 논문들은 본 연구에서 참고한 핵심 이론과 모델 설계 아이디어이다.

[1] A. Beck, P. Stoica, and J. Li, “Exact and Approximate Solutions of Source Localization Problems,” IEEE Transactions on Signal Processing, vol. 56, no. 5, pp. 1770–1778, 2008.

이 논문은 range measurement 또는 range-difference measurement를 이용한 source localization 문제를 least squares 관점에서 다룬다. 본 연구는 거리 측정값과 anchor 좌표로부터 위치를 추정하는 문제 설정과 LS/WLS 기반 위치 계산 관점을 참고하였다. 차이점은 본 연구에서는 anchor weight를 사람이 고정하지 않고 GNN이 학습하며, WLS 결과에 residual correction을 추가한다는 점이다.

[2] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl, “Neural Message Passing for Quantum Chemistry,” Proceedings of the 34th International Conference on Machine Learning, PMLR 70:1263–1272, 2017.

이 논문은 graph node들이 message passing을 통해 표현을 갱신하는 Message Passing Neural Network의 일반적 틀을 제시한다. 본 연구는 이 개념을 분자 그래프가 아니라 RTT 기지국 그래프에 적용하였다. 따라서 각 node는 기지국이고, message passing 결과는 molecular property가 아니라 각 기지국의 reliability score를 계산하는 데 사용된다.

[3] T. N. Kipf and M. Welling, “Semi-Supervised Classification with Graph Convolutional Networks,” ICLR, 2017.

이 논문은 graph-structured data에서 node feature와 graph structure를 함께 이용해 hidden representation을 학습하는 GCN 구조를 제안하였다. 본 연구는 node classification을 수행하지는 않지만, anchor node의 feature와 anchor 간 관계를 함께 반영하여 node representation을 만드는 graph feature aggregation 개념을 참고하였다. 본 연구의 차이점은 graph representation을 분류가 아니라 위치추정을 위한 anchor reliability regression에 사용했다는 점이다.

[4] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016.

이 논문은 residual learning 구조를 제안하였다. 본 연구는 이미지 인식용 ResNet을 그대로 사용한 것이 아니라, WLS가 계산한 초기 위치 $p_{\mathrm{wls}}$를 기준값으로 두고 신경망이 잔차 보정량 $\Delta p$를 학습하도록 응용하였다. 즉 최종 위치를 $p_{\mathrm{final}} = p_{\mathrm{wls}} + \Delta p$로 정의하여, 기하학적 위치 계산으로 설명되지 않는 RTT 오차의 잔차만 보정하도록 설계하였다.
