# Evolutionary-Successive-Halving-Algorithm

## Description

<div align="center">
  <img src="https://github.com/pod3275/Evolutionary-Successive-Halving-Algorithm/blob/master/assets/ESHA.png" width="80%"><br>
</div>

- 기존 하이퍼파라미터 최적화 기법들의 단점을 보완하는 효율적인 하이퍼파라미터 최적화 기법.

- ESHA는 Hyperband의 병렬적인 계산 및 자원을 효율적으로 분배한다는 특성과 Bayesian Optimization의 기존 탐색을 통해 얻은 정보를 활용한다는 특성을 결합한 알고리즘임.

- 이미지 분류 모델을 이용한 실험 결과, ESHA는 총 가용 자원이 낮은 환경에서 기존의 하이퍼파라미터 최적화 기법들에 비해 높은 성능을 보임.

## Requirements
- [Python 3.6](https://www.python.org/downloads/)
- [Tensorflow == 1.12.0](https://github.com/tensorflow/tensorflow)
- [Keras >= 2.2.4](https://github.com/keras-team/keras)
- [Scikit-Optimize](https://scikit-optimize.github.io/) (for Bayesian Optimization)
- [Hyperopt](https://github.com/hyperopt/hyperopt) (for TPE, ESHA)
- [HpBandSter](https://github.com/automl/HpBandSter) (for BOHB)

## Algorithm
- To be continued

## Implemented Methods
- [Bayesian Optimization](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)
- [Hyperband](https://arxiv.org/pdf/1603.06560.pdf)
- [TPE](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)
- [BOHB](https://arxiv.org/pdf/1807.01774.pdf)
- ESHA (Evolutionary Successive Halving Algorithm)
