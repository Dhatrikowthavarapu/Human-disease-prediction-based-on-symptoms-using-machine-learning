#importing the libraries
from tkinter import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']


train=pd.read_csv("Training.csv")
test=pd.read_csv("Testing.csv")
df=pd.concat([train,test])

#Random forest
def RandomForest():
    print("RandomForest")
    X,y=df.iloc[:,:-1], df.iloc[:,-1]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    from sklearn.ensemble import RandomForestClassifier
    rf1 = RandomForestClassifier()
    rf1 = rf1.fit(X_train,y_train)
    y_pred=rf1.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    feature_imp=pd.Series(rf1.feature_importances_,index=list(df.columns[:-1])).sort_values(ascending=False).head(50)
    print(feature_imp[::-1].index)
    X_reduced,y=df[['phlegm', 'red_spots_over_body', 'rusty_sputum', 'belly_pain',
       'spinning_movements', 'dischromic _patches', 'abdominal_pain',
       'loss_of_balance', 'dizziness', 'sunken_eyes', 'extra_marital_contacts',
       'swelled_lymph_nodes', 'bladder_discomfort', 'nodal_skin_eruptions',
       'stomach_pain', 'muscle_weakness', 'neck_pain', 'chills', 'coma',
       'watering_from_eyes', 'abnormal_menstruation', 'loss_of_appetite',
       'continuous_feel_of_urine', 'red_sore_around_nose', 'diarrhoea',
       'malaise', 'nausea', 'yellowish_skin', 'blister', 'headache',
       'movement_stiffness', 'mucoid_sputum', 'ulcers_on_tongue',
       'unsteadiness', 'joint_pain', 'fatigue', 'passage_of_gases', 'sweating',
       'dark_urine', 'weight_loss', 'yellowing_of_eyes', 'altered_sensorium',
       'breathlessness', 'vomiting', 'mild_fever', 'chest_pain', 'high_fever',
       'itching', 'muscle_pain', 'family_history']],df.iloc[:,-1]
    #print(len(X_reduced))
    X_train,X_test,y_train,y_test=train_test_split(X_reduced,y,test_size=0.3)
    #print(y_test)
    rf2=RandomForestClassifier()
    rf2.fit(X_train,y_train)
    y_pred=rf2.predict(X_test)
    a=accuracy_score(y_test, y_pred)
    l2=[]
    l3=['phlegm', 'red_spots_over_body', 'rusty_sputum', 'belly_pain',
       'spinning_movements', 'dischromic _patches', 'abdominal_pain',
       'loss_of_balance', 'dizziness', 'sunken_eyes', 'extra_marital_contacts',
       'swelled_lymph_nodes', 'bladder_discomfort', 'nodal_skin_eruptions',
       'stomach_pain', 'muscle_weakness', 'neck_pain', 'chills', 'coma',
       'watering_from_eyes', 'abnormal_menstruation', 'loss_of_appetite',
       'continuous_feel_of_urine', 'red_sore_around_nose', 'diarrhoea',
       'malaise', 'nausea', 'yellowish_skin', 'blister', 'headache',
       'movement_stiffness', 'mucoid_sputum', 'ulcers_on_tongue',
       'unsteadiness', 'joint_pain', 'fatigue', 'passage_of_gases', 'sweating',
       'dark_urine', 'weight_loss', 'yellowing_of_eyes', 'altered_sensorium',
       'breathlessness', 'vomiting', 'mild_fever', 'chest_pain', 'high_fever',
       'itching', 'muscle_pain', 'family_history']
    for i in range(len(l3)):
        l2.append(0)
    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    
    for k in range(0,len(l3)):
        for z in psymptoms:
            if(z==l3[k]):
                l2[k]=1
    print(l2)
    inputtest=[l2]
    predict=rf2.predict(inputtest)
    predicted=predict[0]
    return a,predicted
    

#RandomForest()

def naive():
    X_reduced,y=df[['phlegm', 'red_spots_over_body', 'rusty_sputum', 'belly_pain',
       'spinning_movements', 'dischromic _patches', 'abdominal_pain',
       'loss_of_balance', 'dizziness', 'sunken_eyes', 'extra_marital_contacts',
       'swelled_lymph_nodes', 'bladder_discomfort', 'nodal_skin_eruptions',
       'stomach_pain', 'muscle_weakness', 'neck_pain', 'chills', 'coma',
       'watering_from_eyes', 'abnormal_menstruation', 'loss_of_appetite',
       'continuous_feel_of_urine', 'red_sore_around_nose', 'diarrhoea',
       'malaise', 'nausea', 'yellowish_skin', 'blister', 'headache',
       'movement_stiffness', 'mucoid_sputum', 'ulcers_on_tongue',
       'unsteadiness', 'joint_pain', 'fatigue', 'passage_of_gases', 'sweating',
       'dark_urine', 'weight_loss', 'yellowing_of_eyes', 'altered_sensorium',
       'breathlessness', 'vomiting', 'mild_fever', 'chest_pain', 'high_fever',
       'itching', 'muscle_pain', 'family_history']],df.iloc[:,-1]
    X_train,X_test,y_train,y_test=train_test_split(X_reduced,y,test_size=0.3)
    from sklearn.naive_bayes import GaussianNB
    gnb=GaussianNB()
    gnb=gnb.fit(X_train,y_train)
    y_pred=gnb.predict(X_test)
    a=accuracy_score(y_test,y_pred)
    l2=[]
    l3=['phlegm', 'red_spots_over_body', 'rusty_sputum', 'belly_pain',
       'spinning_movements', 'dischromic _patches', 'abdominal_pain',
       'loss_of_balance', 'dizziness', 'sunken_eyes', 'extra_marital_contacts',
       'swelled_lymph_nodes', 'bladder_discomfort', 'nodal_skin_eruptions',
       'stomach_pain', 'muscle_weakness', 'neck_pain', 'chills', 'coma',
       'watering_from_eyes', 'abnormal_menstruation', 'loss_of_appetite',
       'continuous_feel_of_urine', 'red_sore_around_nose', 'diarrhoea',
       'malaise', 'nausea', 'yellowish_skin', 'blister', 'headache',
       'movement_stiffness', 'mucoid_sputum', 'ulcers_on_tongue',
       'unsteadiness', 'joint_pain', 'fatigue', 'passage_of_gases', 'sweating',
       'dark_urine', 'weight_loss', 'yellowing_of_eyes', 'altered_sensorium',
       'breathlessness', 'vomiting', 'mild_fever', 'chest_pain', 'high_fever',
       'itching', 'muscle_pain', 'family_history']
    for i in range(len(l3)):
        l2.append(0)
    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    
    for k in range(0,len(l3)):
        for z in psymptoms:
            if(z==l3[k]):
                l2[k]=1
    print(l2)
    inputtest=[l2]
    predict=gnb.predict(inputtest)
    predicted=predict[0]
    return a, predicted

def result():
    rfa,rf=RandomForest()
    nva,nv=naive()
    if(rfa>=nva):
        t1.delete("1.0",END)
        t1.insert(END,rf)
        t2.delete("1.0",END)
        t2.insert(END,rfa)
    else:
        t1.delete("1.0",END)
        t1.insert(END,nv)
        t2.delete("1.0",END)
        t2.insert(END,nva)

l1=['phlegm', 'red_spots_over_body', 'rusty_sputum', 'belly_pain',
       'spinning_movements', 'dischromic _patches', 'abdominal_pain',
       'loss_of_balance', 'dizziness', 'sunken_eyes', 'extra_marital_contacts',
       'swelled_lymph_nodes', 'bladder_discomfort', 'nodal_skin_eruptions',
       'stomach_pain', 'muscle_weakness', 'neck_pain', 'chills', 'coma',
       'watering_from_eyes', 'abnormal_menstruation', 'loss_of_appetite',
       'continuous_feel_of_urine', 'red_sore_around_nose', 'diarrhoea',
       'malaise', 'nausea', 'yellowish_skin', 'blister', 'headache',
       'movement_stiffness', 'mucoid_sputum', 'ulcers_on_tongue',
       'unsteadiness', 'joint_pain', 'fatigue', 'passage_of_gases', 'sweating',
       'dark_urine', 'weight_loss', 'yellowing_of_eyes', 'altered_sensorium',
       'breathlessness', 'vomiting', 'mild_fever', 'chest_pain', 'high_fever',
       'itching', 'muscle_pain', 'family_history']

#Creating the GUI
root=Tk()
root.title("Disease Prediction")
root.configure(background='blue')

#variables
Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)

#Title
w2 = Label(root, justify=LEFT, text="Disease Predictor using Machine Learning", fg="white", bg="white")
w2.config(font=("Elephant", 30))
w2.grid(row=1, column=0, columnspan=2, padx=100)
w2 = Label(root, justify=LEFT, text="", fg="white", bg="white")
w2.config(font=("Aharoni", 30))
w2.grid(row=2, column=0, columnspan=2, padx=100)

#labels
S1Lb = Label(root, text="Symptom 1", fg="yellow", bg="black")
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2", fg="yellow", bg="black")
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="yellow", bg="black")
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="yellow", bg="black")
S4Lb.grid(row=10, column=0, pady=10, sticky=W)

S5Lb = Label(root, text="Symptom 5", fg="yellow", bg="black")
S5Lb.grid(row=11, column=0, pady=10, sticky=W)

empty1Lb = Label(root, text="", fg="white", bg="blue")
empty1Lb.grid(row=15, column=0, pady=10,sticky=W)

RnfLb = Label(root, text="Disease(Predicted)", fg="white", bg="red")
RnfLb.grid(row=17, column=0, pady=10, sticky=W)

empty2Lb = Label(root, text="", fg="white", bg="blue")
empty2Lb.grid(row=19, column=0, pady=10, sticky=W)

RnfLb = Label(root, text="Accuracy", fg="white", bg="red")
RnfLb.grid(row=21, column=0, pady=10, sticky=W)


OPTIONS = sorted(l1)
print(l1)

S1En = OptionMenu(root, Symptom1,*OPTIONS)
S1En.grid(row=7, column=1)

S2En = OptionMenu(root, Symptom2,*OPTIONS)
S2En.grid(row=8, column=1)

S3En = OptionMenu(root, Symptom3,*OPTIONS)
S3En.grid(row=9, column=1)

S4En = OptionMenu(root, Symptom4,*OPTIONS)
S4En.grid(row=10, column=1)

S5En = OptionMenu(root, Symptom5,*OPTIONS)
S5En.grid(row=11, column=1)


rnf = Button(root, text="Predict", command=result,bg="green",fg="yellow")
rnf.grid(row=9, column=3,padx=10)

#nbb = Button(root, text="Predict2", command=nb,bg="green",fg="yellow")
#nbb.grid(row=11, column=3,padx=10)

t1 = Text(root, height=1, width=40,bg="orange",fg="black")
t1.grid(row=17, column=1 , padx=10)

t2 = Text(root, height=1, width=40,bg="orange",fg="black")
t2.grid(row=21, column=1 , padx=10)

root.mainloop()
