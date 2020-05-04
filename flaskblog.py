from flask import Flask, render_template, url_for
app = Flask(__name__)
from sqlcreater import showdata

inst = showdata()
posthome = (inst.showresult(All=True))
DictA, DictC, DictS, DictO,DictCOMB, DictBOHREN,DictConditioning,Performance,Powersave,TEP = inst.showresult(All=False)



@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', posthome=DictCOMB)


@app.route("/seperate")
def seperate():
    return render_template('seperate.html', DictA =DictA , DictC = DictC, DictS=DictS , DictO = DictO)

@app.route("/ReadMe")
def ReadMe():
    return render_template(('ReadMe.html'))
@app.route("/seperate2")
def seperate2():
    return render_template('seperate2.html',DictBOHREN =DictBOHREN, DictConditioning = DictConditioning,Performance=Performance,Powersave=Powersave,TEP=TEP)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000', debug=True)