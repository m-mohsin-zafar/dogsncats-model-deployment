import os
from flask import Flask, render_template, request
from inference import get_prediction
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('File not Uploaded..!')
            return
        file = request.files['file']
        img = file.read()
        pet_class, pet_name = get_prediction(image_bytes=img)
        return render_template('results.html', class_id=pet_class, pet=pet_name)


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv('PORT', 5000))
