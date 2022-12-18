demo.py
import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import filedialog
import time
import pandas as pd

import stock_predict as pred


def creat_windows():
    win = tk.Tk()
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    ww, wh = 800, 450
    x, y = (sw - ww) / 2, (sh - wh) / 2
    win.geometry("%dx%d+%d+%d" % (ww, wh, x, y - 40)) 

    win.title('LSTM Stock Prediction') 

    f_open =open('AAP.csv')
    canvas = tk.Label(win)
    canvas.pack()

    var = tk.StringVar() 
    var.set('Choosing Dataset')
    tk.Label(win, textvariable=var, bg='#C1FFC1', font=('宋体', 21), width=20, height=2).pack()

    tk.Button(win, text='Choosing Dataset', width=20, height=2, bg='#FF8C00', command=lambda: getdata(var, canvas),
              font=('circle', 10)).pack()

    canvas = tk.Label(win)
    L1 = tk.Label(win, text="Chooing the row tha you need(Please separate them by space from 0）")
    L1.pack()
    E1 = tk.Entry(win, bd=5)
    E1.pack()
    button1 = tk.Button(win, text="Submit", command=lambda: getLable(E1))
    button1.pack()
    canvas.pack()
    win.mainloop()

def getLable(E1):
    string = E1.get()
    print(string)
    gettraindata(string)

def getdata(var, canvas):
    global file_path
    file_path = filedialog.askopenfilename()
    var.set("The last one is label")
    with open(file_path, 'r', encoding='gb2312') as f:
    # with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines() 
        data2 = lines[0]
    print()

    canvas.configure(text=data2)
    canvas.text = data2

def gettraindata(string):
    f_open = open(file_path)
    df = pd.read_csv(f_open) 
    list = string.split()
    print(list)
    x = len(list)
    index=[]
    # data = df.iloc[:, [1,2,3]].values  
    for i in range(x):
        q = int(list[i])
        index.append(q)
    global data
    data = df.iloc[:, index].values
    print(data)
    main(data)

def main(data):
    pred.LSTMtest(data)
    var.set("The prediction is：" + answer)

if __name__ == "__main__":
    creat_windows()
