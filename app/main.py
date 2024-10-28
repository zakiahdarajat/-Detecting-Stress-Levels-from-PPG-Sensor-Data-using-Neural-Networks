from flask import Flask, render_template, request
import model.model as m
import numpy as np

app = Flask(__name__)
label = ['condition']
features = ['MEAN_RR','RMSSD','pNN25','pNN50','LF','HF','LF_HF']
st=""

@app.route("/", methods = ["GET","POST"])
def hello():
    l=[]
    if request.method == "POST":
        for i in features:
            try:
                l.append(float(request.form[i]))
            except ValueError:
                l.append(request.form[i])
        p = m.predict_pipe(l)
        p = np.argmax(p[0])

        if p == 0:
            st = "No Stress"
        elif p == 1:
            st = "Low Stress"
        else:
            st = "High"
    else:
        st=""



    return render_template('index.html', cond=st)




if __name__ == "__main__":
    app.run(debug=True)