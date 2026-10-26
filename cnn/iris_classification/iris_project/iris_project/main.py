"""이 파일만 실행하세요. --model 인자로 네 가지 모델 중 하나를 선택합니다."""
import argparse


def main():
    parser = argparse.ArgumentParser(description='Iris 네 가지 모델 비교 실습')
    parser.add_argument('--model', choices=['keras-sequential', 'keras-functional',
                        'torch-sequential', 'torch-forward'], default='keras-sequential')
    parser.add_argument('--csv', default=None, help='CSV 파일 경로 또는 URL (생략하면 GitHub)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or not 0 < args.lr < float('inf'):
        parser.error('epochs >= 1, batch-size >= 2, 유한한 lr > 0이 필요합니다.')

    #################################################
    # 전처리 데이터 import
    #################################################
    from iris_data import IrisData
    data = IrisData() if args.csv is None else IrisData(args.csv)
    print('입력 특성 순서:', data.feature_columns)
    print('학습 입력/정답:', data.X_train.shape, data.y_train.shape)
    print('테스트 입력/정답:', data.X_test.shape, data.y_test.shape)

    #################################################
    # 만들어둔 모델/실행 파일 import하기.
    # 그 파일에서 선택한 라이브러리만 import하므로 다른 라이브러리는 설치하지 않아도 됩니다.
    #################################################
    if args.model.startswith('keras'):
        import tensorflow as tf
        import keras_models as models
        import keras_runner as runner
        tf.keras.utils.set_random_seed(42)
        model = (models.build_sequential() if args.model == 'keras-sequential'
                 else models.build_functional())
        model.summary()
    else:
        import torch
        import torch_models as models
        import torch_runner as runner
        torch.manual_seed(42)
        model = (models.build_sequential() if args.model == 'torch-sequential'
                 else models.IrisModel())
        print(model)

    runner.train(model, data, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    result = runner.evaluate(model, data)
    print(f"테스트 loss: {result['loss']:.4f}, accuracy: {result['accuracy']:.2%}")
    rows = [[5.1, 3.5, 1.4, 0.2], [6.0, 2.9, 4.5, 1.5], [6.5, 3.0, 5.8, 2.2]]
    names, probabilities = runner.predict(model, data, rows)
    print('확률 열 순서:', data.class_names.tolist())
    for row, name, probs in zip(rows, names, probabilities):
        print(f'입력 {row} → 예측 {name}, 확률 {probs.round(4)}')


if __name__ == '__main__':
    main()
