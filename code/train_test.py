import argparse
from build_model import ModelBuilder
import config
from keras.utils import image_dataset_from_directory, img_to_array, load_img
from sklearn.metrics import confusion_matrix
from keras.models import load_model, save_model
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask Detector')
    parser.add_argument('-i', '--input', type=str, help='Path to input image')
    parser.add_argument('-m', '--mode', type=str, help='Choose mode: train or test')

    args = vars(parser.parse_args())


    if args['input'] is None:
        parser.print_help()
    else:
        (train_data, validation_data) = image_dataset_from_directory(
            args['input'],
            labels="inferred",
            color_mode="rgb",
            batch_size=32,
            image_size=config.IMAGE_SIZE,
            shuffle=True,
            seed=config.RANDOM_SEED,
            validation_split=0.2,
            subset='both',
        )
        if args['mode'] not in ['train', 'test']:
            print('Invalid mode')
            raise Exception("Wrong mode, choose train or test")
        elif args['mode'] == 'train':
            # building model
            model = ModelBuilder(config.INPUT_SHAPE, config.NUM_CLASSES)
            model = model.build_model()
            model.fit(train_data,
                      callbacks=config.CALLBACKS,
                      epochs=config.EPOCHS,
                      batch_size=config.BATCH_SIZE,
                      validation_data=validation_data)
        elif args['mode'] == 'test':
            # loading and evaluating model
            model = load_model(config.MODEL_PATH)
            predictions = model.predict(validation_data)
            y_pred = np.argmax(predictions, axis=1)
            tmp = np.array([])
            for i, l in validation_data:
                tmp = np.hstack([tmp, np.array(l)])
            print(confusion_matrix(tmp, y_pred))
            print(model.evaluate(validation_data))


