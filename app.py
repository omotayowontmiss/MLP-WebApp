from flask import Flask, render_template, request
from mlp_model import predict

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        values = []
        for i in range(10):
            values.append(float(request.form[f'feature{i}']))

        prediction = predict(values)

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run()
