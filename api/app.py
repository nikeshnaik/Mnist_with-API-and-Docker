try:
    import unzip_requirements  # pylint: disable=unused-import
except ImportError:
    pass

from flask import Flask, request, jsonify
import tensorflow as tf

from digit_recognizer.digit_predictor import DigitPredictor

import digit_recognizer.utils as util


app = Flask(__name__)
# with tf.Session().graph.as_default() as _:
predictor = DigitPredictor()


@app.route('/')
def index():
    return "hello WOrld"

@app.route('/v1/predict',methods=['GET','POST'])
def predict():
    image = _load_image()
    # with tf.Session().graph.as_default() as _:
    pred, conf = predictor.predict(image)
    print("METRIC confidence {}".format(conf))
    print("METRIC mean_intensity {}".format(image.mean()))
    print("INFO pred {}".format(pred))
    return jsonify({'pred': str(pred), 'conf': float(conf)})

def _load_image():
    if request.method == 'POST':
        data = request.get_json()
        if data is None:
            return 'no json received'
        return util.read_b64_image(data['image'], grayscale=True)
    if request.method == 'GET':
        image_url = request.args.get('image_url')
        if image_url is None:
            return 'no image_url defined in query string'
        print("INFO url {}".format(image_url))
        return util.read_image(image_url, grayscale=True)
    raise ValueError('Unsupported HTTP method')

def main():
    app.run(host='0.0.0.0', port=8000, debug=False)  # nosec


if __name__ == '__main__':
    main()
