from flask import Flask, render_template, request
import model.model as m
import numpy as np

app = Flask(__name__)
label = ['condition']
features = ['MEAN_RR','RMSSD','pNN25','pNN50','LF','HF','LF_HF']

@app.route("/", methods = ["GET","POST"])
def hello():
    l=[]
    st = ""  # Initialize locally to avoid global variable issues
    if request.method == "POST":
        try:
            for i in features:
                value = request.form.get(i, '')
                if not value:
                    st = "Error: Missing required field: " + i
                    return render_template('index.html', cond=st)
                l.append(float(value))
            p = m.predict_pipe(l)
            p = np.argmax(p[0])
            
            if p == 0:
                st = "No Stress"
            elif p == 1:
                st = "Low Stress"
            else:
                st = "High Stress"  # Fixed: Complete the stress level description
        except (ValueError, TypeError) as e:
            st = "Error: Invalid input. Please enter valid numeric values for all fields."
            return render_template('index.html', cond=st)
        except Exception as e:
            st = "Error: Prediction failed. Please try again."
            return render_template('index.html', cond=st)



    return render_template('index.html', cond=st)




if __name__ == "__main__":
    import os
    # Security fix: Only enable debug mode in development
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, host='127.0.0.1')