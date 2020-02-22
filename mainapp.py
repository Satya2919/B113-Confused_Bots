from flask import Flask,render_template,request
import sqlite3 as sql
import numpy as np
import pickle
model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)
id=""
@app.route('/')
def website():
    return render_template("login.html")
@app.route('/reciptionist')
def rep():
    con = sql.connect("hackoverflow.db")
    con.row_factory = sql.Row
    cur = con.cursor()
    cur.execute("select * from apointments")
    rows=cur.fetchall()
    con.close()
    return render_template("reciptionist.html",rows=rows)
@app.route('/recupload',methods=['POST','GET'])
def recp():
    if request.method == 'POST':
        try:
            pn=request.form['PN']
            casse = request.form['case']
            condition = request.form['condition']
            print("-----------------"+pn)
            with sql.connect("hackoverflow.db") as con:
                cur = con.cursor()
                rr=cur.execute("SELECT patient_id,name,age from patientdetails WHERE phonenumber="+(pn))
            for row in rr:
                print(row[0],row[1],row[2])
                pid=row[0]
                nm=row[1]
                ag=row[2]
            insert(pid,nm,ag,casse,condition)
        except:
            con.rollback()
            msg = "error in insert operation"
        finally:
            con = sql.connect("hackoverflow.db")
            con.row_factory = sql.Row
            cur = con.cursor()
            cur.execute("select * from apointments")
            rows=cur.fetchall();
            con.commit()
            cur.close()
            return render_template("reciptionist.html",rows=rows,msg=id)
def insert(pid,nm,ag,casse,condition):
    print(pid,nm,ag,casse,condition)
    try:
        with sql.connect("hackoverflow.db") as con:
                    cur=con.cursor()
                    cur.execute("INSERT INTO apointments (patient_id,name,age,caseof,condition) VALUES(?,?,?,?,?)",(str(pid),nm,ag,casse,condition))
                    con.commit()
    except:
        con.rollback()
@app.route('/log' ,methods=['POST','GET'])
def log():
    if request.method == 'POST':
        try:
            rol= request.form['rol']
            nm = request.form['name']
            pas = request.form['password']
            print(nm+"====="+pas)
            if rol=="patient":
                with sql.connect("hackoverflow.db") as con:
                    cur = con.cursor()
                    cur.execute("SELECT userid,password FROM user_login")
                    info=cur.fetchall()
                    f=0
                    for row in info:
                        print(row[0],row[1])
                        if nm==row[0] and pas==row[1]:
                            return render_template("patient.html")
            with sql.connect("hackoverflow.db") as con:
                cur = con.cursor()
                cur.execute("SELECT userid,password,role FROM user_login")
                info=cur.fetchall()
                f=0
                for row in info:
                    print(row[0],row[1])
                    if nm==row[0] and pas==row[1] and rol==row[2]:
                        if rol=="doctor":
                            f = f + 1
                            return render_template("index.html")
                            break
                        elif rol=="Receptionist":
                            f = f + 1
                            return render_template("reciptionist.html")
                            break
                if f==0:
                    msg="invalid id or password"
                    return render_template("login.html",msg=msg)
                con.commit()
        except:
            con.rollback()
            msg = "internal error"
            return render_template("login.html")
@app.route('/index.html')
def index():
    with sql.connect("hackoverflow.db") as con:
        con.row_factory = sql.Row
        cur = con.cursor()
        cur.execute("select * from apointments")
        rows = cur.fetchall();
        return render_template("index.html", rows=rows,id=id)
    return render_template("index.html")
@app.route('/tables.html',methods=['POST','GET'])
def search():
    id=request.form['id']
    with sql.connect("formdata.db") as con:
        con = sql.connect("formdata.db")
        con.row_factory = sql.Row
        cur = con.cursor()
        cur.execute("select * from students")
        rows = cur.fetchall();
        return render_template("tables.html", rows=rows,id=id)

@app.route('/upload' , methods=['POST','GET'])
def formdata():
    if request.method == 'POST':
        try:
            id=request.form['id']
            heartrate = request.form['HR']
            ppulse = request.form['pulse']
            hhemoglobin = request.form['hemoglobin']
            aage = request.form['age']
            ffever = request.form['fever']
            ooc2 = request.form['OC2']
            with sql.connect("formdata.db") as con:
                cur = con.cursor()
                cur.execute("INSERT INTO students (patient_id,heart_rate, pulse,hemoglobin,age,fever,oc2) VALUES(?,?,?,?,?,?,?)", (id,heartrate, ppulse,hhemoglobin,aage,ffever,ooc2))
                con.commit()
                percentage=int(float(float(ooc2)/1000)*100)
                print (float(float(ooc2)/1000))
                print(ooc2)
                print(percentage)
                msg = "percentage of having sepsis:"+str(percentage)+"%"
                print("successs======================")

        except:
            con.rollback()
            msg = "error in insert operation"
        finally:
            con = sql.connect("formdata.db")
            con.row_factory = sql.Row
            cur = con.cursor()
            cur.execute("select * from students")
            rows=cur.fetchall();
            return render_template("tables.html",rows=rows,msg=msg)
            con.close()

@app.route('/predict',methods=['POST'])
def predict():
    hr=request.form['HR']
    pul=request.form['pulse']
    hemoglobin=request.form['hemoglobin']
    age=request.form['age']
    fever=request.form['fever']
    oc2=request.form['OC2']
    int_features=[int(hr),int(pul),int(hemoglobin),int(age),int(fever),int(oc2)]
    print(int_features)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('tables.html', prediction_text=(prediction))
if __name__ == '__main__':

    app.run(debug=True)
