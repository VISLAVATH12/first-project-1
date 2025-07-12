from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import model_selection,neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns



main = tk.Tk()
main.title("Flood Forecasting Using ML")
main.geometry("1600x1500")

font = ('times', 16, 'bold')
title = Label(main, text='FLOOD FORECASTING USING MACHINE LEARNING',font=("times"))
title.config(bg='Dark Blue', fg='white')
title.config(font=font)
title.config(height=3, width=105)
title.place(x=0, y=5)

global x,y,x_train,y_train,x_test,y_test

font1 = ('times', 12, 'bold')
text = Text(main, height=14, width=110)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=150, y=120)
text.config(font=font1)

def upload():
    global filename, df
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    df = pd.read_csv(filename)
    text.delete('1.0', END)
    text.insert(END, 'Dataset loaded\n')
    text.insert(END, "Dataset Size: " + str(len(df)) + "\n")


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload File",command=upload,width=20)
uploadButton.place(x=200, y=400)
uploadButton.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')
pathlabel.config(font=font1)
pathlabel.place(x=450, y=400)

def split1():
    global x,y,x_train, x_test, y_train, y_test
    x = df.iloc[:, 1:14]
    y = df['FLOODS']
    print(x)
    print(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    text.delete('1.0', END)
    text.insert(END, str(x))
    text.insert(END, str(y))
    return x, y, x_train, x_test, y_train, y_test

splitButton = Button(main, text="Split Dataset",command=split1,width=20)
splitButton.place(x=50, y=470)
splitButton.config(font=font1)

def create_histogram_popup():
    popup = tk.Toplevel(main)
    popup.title("Histogram")

    canvas = FigureCanvasTkAgg(plt.gcf(), master=popup)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()



    ax = df[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].mean().plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2,figsize=(14, 6))
    plt.xlabel('Month', fontsize=10)
    plt.ylabel('Monthly Rainfall', fontsize=10)
    plt.title('Rainfall in Kerela for all Months', fontsize=10)
    ax.tick_params(labelsize=20)
    plt.grid()
    plt.ioff()


dtButton = Button(main, text="Generate Histogram", command=create_histogram_popup,width=20)
dtButton.place(x=900, y=700)
dtButton.config(font=font1)

def knn_algorithm():
    text.delete('1.0', END)

    text.insert(END,"KNN ACCURACY "+ "\n\n")
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(x_train, y_train)
    y_pred = knn_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print( f"Accuracy of KNN is: {accuracy * 100:.2f}%" + "\n\n")
    text.insert(END, f"Accuracy of KNN is: {accuracy * 100:.2f}%" + "\n\n")
    return knn_classifier


open_second_button = tk.Button(main,font=(13), text="Run KNN Algorithm",command=knn_algorithm,width=20)
open_second_button.place(x=350, y=470)
open_second_button.config(font=font1)


def logistic():
    global logistic_classifier,accuracy1

    text.delete('1.0', END)
    text.insert(END,"Logistic ACCURACY "+ "\n\n")
    logistic_classifier = LogisticRegression()
    logistic_classifier.fit(x_train, y_train)
    y_pred = logistic_classifier.predict(x_test)
    accuracy1 = accuracy_score(y_test, y_pred)
    print(f"Accuracy of Logistic Regression is: {accuracy1 * 100:.2f}%" + "\n\n")
    text.insert(END, f"Accuracy of Logistic Regression is: {accuracy1 * 100:.2f}%" + "\n\n")
    return logistic_classifier



open_second_button = tk.Button(main,font=(13), text="Run Logistic Regression",command=logistic,width=20)
open_second_button.place(x=600, y=470)
open_second_button.config(font=font1)


def svm1():
    text.delete('1.0', END)
    text.insert(END, "SVM ACCURACY " + "\n\n")
    svm_classifier = SVC(kernel='rbf', probability=True)
    svm_classifier.fit(x_train, y_train)
    y_pred = svm_classifier.predict(x_test)
    accuracy2 = accuracy_score(y_test, y_pred)
    print(f"Accuracy of SVM is: {accuracy2 * 100:.2f}%" + "\n\n")
    text.insert(END, f"Accuracy of SVM is: {accuracy2 * 100:.2f}%" + "\n\n")
    return svm_classifier


splitButton = Button(main, text="Run svm1",command=svm1,width=20)
splitButton.place(x=850, y=470)
splitButton.config(font=font1)

def dt():
    text.delete('1.0', END)
    text.insert(END, "Decition Tree ACCURACY " + "\n\n")
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(x_train, y_train)
    y_pred = dt_classifier.predict(x_test)
    accuracy3 = accuracy_score(y_test, y_pred)
    print(f"Accuracy of Decision Tree is: {accuracy3 * 100:.2f}%" + "\n\n")
    text.insert(END, f"Accuracy of Decision Tree is: {accuracy3 * 100:.2f}%" + "\n\n")
    return dt_classifier




splitButton = Button(main, text="Run Decision Tree",command=dt,width=20)
splitButton.place(x=1100, y=470)
splitButton.config(font=font1)

def rf():
    global rf_classifier,accuracy4
    text.delete('1.0', END)
    text.insert(END, "Random Forest ACCURACY " + "\n\n")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(x_train, y_train)
    y_pred = rf_classifier.predict(x_test)
    accuracy4 = accuracy_score(y_test, y_pred)
    print(f"Accuracy of Random Forest is: {accuracy4 * 100:.2f}%" + "\n\n")
    text.insert(END, f"Accuracy of Random Forest is: {accuracy4 * 100:.2f}%" + "\n\n")
    return rf_classifier


splitButton = Button(main, text="run random forest",command=rf,width=20)
splitButton.place(x=100, y=550)
splitButton.config(font=font1)

def table1():
    global tr_split
    text.delete('1.0', END)
    text.insert(END, "All Algorithms Results " + "\n\n")
    models = []
    models.append(('KNN :',knn_algorithm()))
    models.append(('LR :', logistic()))
    models.append(('SVC :', svm1()))
    models.append(('DT :', dt()))
    models.append(('RF :', rf()))
    names = []
    scores = []
    for name, model in models:
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        scores.append(accuracy)
        names.append(name)
    tr_split = pd.DataFrame({'Name': names, 'Score': scores})
    print("All Algorithms Result: " + str(tr_split))
    text.insert(END, str(tr_split))
    return names, scores

splitButton = Button(main, text="Algorithms",command=table1,width=20)
splitButton.place(x=400, y=550)
splitButton.config(font=font1)


def show_algorithm_results():
    names, scores = table1()
    graph_window = tk.Toplevel(main)
    graph_window.title("Algorithm Results")
    plt.figure(figsize=(8, 6))
    plt.bar(names, scores, color='skyblue')
    plt.ylabel('Accuracy Score')
    plt.title('Algorithm Accuracy Comparison')

    for i, score in enumerate(scores):
        plt.text(i, score, f'{score:.2f}', ha='center', va='bottom', fontsize=10, color='black')

    plt.xticks(rotation=45, ha="right")
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

    canvas = FigureCanvasTkAgg(plt.gcf(), master=graph_window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

show_results_button = tk.Button(main, font=(13), text="Show Algorithm Results", command=show_algorithm_results,width=20)
show_results_button.place(x=650, y=550)
show_results_button.config(font=font1)

def open_second_page():
    second_window = tk.Toplevel(main)
    second_window.title("Second Page")
    second_window.geometry("700x900")
    second_window.config(bg='#FFDAB9')

    label1 = tk.Label(second_window, font=("times", 15), text="Enter The Values For Prediction")
    label1.pack(pady=10)

    # Add 13 labels for input fields
    label_year = tk.Label(second_window, font=("times", 15), text="Year:")
    label_year.place(x=200,y=50)

    input_year = tk.Entry(second_window, font=("times", 15))
    input_year.place(x=300,y=50)

    label_january = tk.Label(second_window, font=("times", 15), text="January:")
    label_january.place(x=200, y=85)

    input_january = tk.Entry(second_window, font=("times", 15))
    input_january.place(x=300, y=85)

    label_February = tk.Label(second_window, font=("times", 15), text="February:")
    label_February.place(x=200, y=120)

    input_February = tk.Entry(second_window, font=("times", 15))
    input_February.place(x=300, y=120)

    label_March = tk.Label(second_window, font=("times", 15), text="March:")
    label_March.place(x=200, y=155)

    input_march = tk.Entry(second_window, font=("times", 15))
    input_march.place(x=300, y=155)

    label_april = tk.Label(second_window, font=("times", 15), text="April:")
    label_april.place(x=200, y=190)

    input_april = tk.Entry(second_window, font=("times", 15))
    input_april.place(x=300, y=190)

    label_may = tk.Label(second_window, font=("times", 15), text="May:")
    label_may.place(x=200, y=225)

    input_may = tk.Entry(second_window, font=("times", 15))
    input_may.place(x=300, y=225)

    label_june = tk.Label(second_window, font=("times", 15), text="June:")
    label_june.place(x=200, y=260)

    input_june = tk.Entry(second_window, font=("times", 15))
    input_june.place(x=300, y=260)

    label_july = tk.Label(second_window, font=("times", 15), text="July:")
    label_july.place(x=200, y=295)

    input_july = tk.Entry(second_window, font=("times", 15))
    input_july.place(x=300, y=295)

    label_aguest = tk.Label(second_window, font=("times", 15), text="Aguest:")
    label_aguest.place(x=200, y=330)

    input_aguest = tk.Entry(second_window, font=("times", 15))
    input_aguest.place(x=300, y=330)

    label_september = tk.Label(second_window, font=("times", 15), text="September:")
    label_september.place(x=200, y=365)

    input_september = tk.Entry(second_window, font=("times", 15))
    input_september.place(x=300, y=365)

    label_october = tk.Label(second_window, font=("times", 15), text="October:")
    label_october.place(x=200, y=400)

    input_october = tk.Entry(second_window, font=("times", 15))
    input_october.place(x=300, y=400)

    label_november = tk.Label(second_window, font=("times", 15), text="November:")
    label_november.place(x=200, y=435)

    input_november = tk.Entry(second_window, font=("times", 15))
    input_november.place(x=300, y=435)

    label_december = tk.Label(second_window, font=("times", 15), text="December:")
    label_december.place(x=200, y=470)

    input_december = tk.Entry(second_window, font=("times", 15))
    input_december.place(x=300, y=470)



    def submit_second_page():
        global input_data, year, january, february, march, april, may, june, july, august, september, october, november, december

        year = input_year.get()
        january = input_january.get()
        february = input_February.get()
        march = input_march.get()
        april = input_april.get()
        may = input_may.get()
        june = input_june.get()
        july = input_july.get()
        august = input_aguest.get()
        september = input_september.get()
        october = input_october.get()
        november = input_november.get()
        december = input_december.get()

        input_data = pd.DataFrame({
            'Year': [year],
            'January': [january],
            'February': [february],
            'March': [march],
            'April': [april],
            'May': [may],
            'June': [june],
            'july': [july],
            'aguest': [august],
            'september': [september],
            'october':[october],
            'november': [november],
            'december':[december]


        })
        text.delete('1.0', END)
        text.insert(END, input_data)
        print(input_data)

        second_window.destroy()

    submit_button = tk.Button(second_window, text="Submit", command=submit_second_page, bg="turquoise", width=10)
    submit_button.place(x=300,y=550)

open_second_button = tk.Button(main,font=(13), text="Enter The Values For Predition", command= open_second_page)
open_second_button.place(x=900, y=550)
open_second_button.config(font=font1)


def predict():
    l1=LabelEncoder()
    l1.fit(y_train)

    records = input_data.values[:, 0:13]
    print("===>", records)
    value = logistic_classifier.predict(records)
    print("result of Logistic Regression is :" + str(value))
    text.insert(END,"\n\n")
    text.insert(END, "Result Of Logistic Regression is : " + str(value) + "\n\n")


# Predict Button
open_second_button = tk.Button(main,font=(13), text="Predition", command=predict)
open_second_button.place(x=650, y=600)
open_second_button.config(font=font1)

main.config(bg='#F08080')
main.mainloop()