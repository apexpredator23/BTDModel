from tabnanny import verbose
from urllib import response
import numpy as np
from flask import Flask,render_template,request,url_for,flash
import tensorflow as tf
import os
import shutil
from flask import send_from_directory

# from tf.keras.preprocessing.image import ImageDataGenerator
# import numpy as np
myapp=Flask(__name__)
myapp.secret_key="super seceret key"

session = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()

model = tf.keras.models.load_model('./BrainTumorPredictionModel.h5', compile=False)

try:
    shutil.rmtree('uploaded')
except:
    pass

os.mkdir("uploaded")

myapp.config['IMAGE_UPLOADS'] = 'uploaded'

@myapp.route('/')
def welcome():
    return render_template("btsa.html")

def finds():
    img = tf.keras.preprocessing.image.load_img(
      "./uploaded/test.jpg", target_size=(180, 180)
  )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 
# Create a batch
    class_names=['No The MRI image is not tumorous.','Yes. The MRI image is Tumorous.']

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    pred = class_names[np.argmax(score)]
    score1 = 100 * np.max(score)

    return pred
    


@myapp.route('/upload-image',methods=['POST'])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["i1"]
            image.save(os.path.join(myapp.config["IMAGE_UPLOADS"], "test.jpg"))
            val=finds()
            return render_template("predictions.html", result=val)
    # return render_template("upload_image.html")
# def predict():
#     if request.method == 'POST':
#         f = request.files['file']
#         f.save(os.path.join("/uploaded/image", secure_filename(f.filename)))
#         val = finds()
#         return render_template('predictions.html', result = val)

    # if(request.method=="POST"):
    #     image = request.files['i1']
    #     print(image)
    #     img=tf.keras.preprocessing.image.load_img("./y29.jpg",target_size=(180,180))
        # img_array=tf.keras.preprocessing.image.img_to_array(img)
    #     img_array=tf.expand_dims(img_array,0)
    #     # model=tf.keras.models.load_model('BrainTumorPredictionModel.h5')
    #     # class_names=['no','yes']
    #     with graph.as_default():
    #         tf.compat.v1.enable_eager_execution()
    #         tf.compat.v1.keras.backend.set_session(session)
    #         rs = model.predict(img_array,steps=1,verbose=1)
    #         print(rs)
    #     # prediction=model.predict(img_array)
    #     score=tf.nn.softmax(rs[0])
    #     return render_template("predictions.html",result=score)

        # pred=class_names[np.argmax(score)]


@myapp.route('/register')
def register():
    return render_template("login.html")
        
@myapp.route('/login')
def login():
    return render_template("BTS.html")

@myapp.route('/imageUpload')
def imageUpload():
    return render_template("imageUpload.html")


@myapp.route('/uploads/<filename>')
def send_uploaded_file(filename=''):
    return send_from_directory(myapp.config["IMAGE_UPLOADS"], filename)



if __name__=="__main__":
    myapp.run(debug=True)