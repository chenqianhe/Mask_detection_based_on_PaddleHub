# coding: utf8
import requests
import json
import base64
import os
import tkinter.filedialog
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox
import sys


rpath = "test.png"
def callbackClose():
    tkinter.messagebox.showwarning(title='警告', message='点击了关闭按钮')
    sys.exit(0)


def selectPath():
    global rpath
    # 选择文件path_接收文件地址
    path_ = tkinter.filedialog.askopenfilename()

    # 通过replace函数替换绝对文件地址中的/来使文件可被程序读取
    # 注意：\\转义后为\，所以\\\\转义后为\\
    path_ = path_.replace("/", "\\\\")
    rpath = path_
    # path设置path_的值
    path.set(path_)


def begin():  # 当acction被点击时,该函数则生效

    global rpath
    action1.configure(text='已上传文件')  # 设置button显示的内容

    action1.configure(state='disabled')  # 将按钮设置为灰色状态，不可使用状态

    # 指定要检测的图片并生成列表[("image", img_1), ("image", img_2), ... ]
    file_list = [rpath]
    files = [("image", (open(item, "rb"))) for item in file_list]
    # 指定检测方法为pyramidbox_lite_server_mask并发送post请求
    url = "http://39.97.120.36:8860/predict/image/pyramidbox_lite_server_mask"
    # url = "http://127.0.0.1:8866/predict/image/pyramidbox_lite_server_mask"
    #
    r = requests.post(url=url, files=files)
    results = eval(r.json()["results"])
    # 保存检测生成的图片到output文件夹，打印模型输出结果
    if not os.path.exists("output"):
        os.mkdir("output")
    for item in results:
        with open(os.path.join("output", item["path"]), "wb") as fp:
            fp.write(base64.b64decode(item["base64"].split(',')[-1]))
            item.pop("base64")
    print(json.dumps(results, indent=4, ensure_ascii=False))

    tkinter.messagebox.showwarning(title='提示', message='已保存到output文件夹')




main_box = tk.Tk()
#变量path
path = tk.StringVar()

main_box.title("口罩佩戴分类系统")  # 添加标题
tk.Label(main_box, text="目标路径:").grid(row=0, column=0)

ttk.Label(main_box, text="请保证路径不含中文").grid(column=1, row=1)
tk.Entry(main_box, textvariable = path).grid(row=0, column=1)
tk.Button(main_box, text="路径选择", command=selectPath).grid(row=0, column=2)

main_box.protocol("WM_DELETE_WINDOW", callbackClose)

action1 = ttk.Button(main_box, text="开始", command=begin)  # 创建一个按钮, text：显示按钮上面显示的文字, command：当这个按钮被点击之后会调用command函数

action1.grid(column=2, row=1)  # 设置其在界面中出现的位置 column代表列 row 代表行

main_box.mainloop()




