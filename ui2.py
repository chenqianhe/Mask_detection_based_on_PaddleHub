# -*- coding: utf-8 -*-
# 
import os
os.environ['HUB_HOME'] = "./modules"
import sys
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox
import paddlehub as hub
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import wx
import wx.xrc
import warnings
warnings.filterwarnings("ignore")



class MyFrame1(wx.Frame):

	def __init__(self, parent):
		wx.Frame.__init__(self, parent, id=wx.ID_ANY, title="口罩佩戴视频监测系统", pos=wx.DefaultPosition,
						  size=wx.Size(552, 121), style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)

		self.SetSizeHintsSz(wx.DefaultSize, wx.DefaultSize)
		self.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_ACTIVECAPTION))
		self.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_ACTIVECAPTION))

		gSizer1 = wx.GridSizer(0, 2, 0, 0)

		self.m_button3 = wx.Button(self, wx.ID_ANY, u"开始", wx.DefaultPosition, wx.DefaultSize, 0)
		self.m_button3.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_INFOTEXT))
		self.m_button3.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_SCROLLBAR))

		gSizer1.Add(self.m_button3, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL, 5)

		self.m_button4 = wx.Button(self, wx.ID_ANY, u"终止", wx.DefaultPosition, wx.DefaultSize, 0)
		self.m_button4.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))
		self.m_button4.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_SCROLLBAR))

		gSizer1.Add(self.m_button4, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL, 5)

		self.SetSizer(gSizer1)
		self.Layout()

		self.Centre(wx.BOTH)

		# Connect Events
		self.Bind(wx.EVT_CLOSE, self.MyFrame1OnClose)
		self.m_button3.Bind(wx.EVT_BUTTON, self.m_button3OnButtonClick)
		self.m_button4.Bind(wx.EVT_BUTTON, self.m_button4OnButtonClick)

	def __del__(self):
		pass


	# Virtual event handlers, overide them in your derived class
	def MyFrame1OnClose( self, event ):
		event.Skip()
		sys.exit(0)

	# Virtual event handlers, overide them in your derived class
	def m_button3OnButtonClick(self, event):
		event.Skip()
		win = tk.Tk()

		win.title("口罩佩戴视频监测系统")  # 添加标题

		ttk.Label(win, text="选择摄像头（默认为0）").grid(column=0, row=0)  # 添加一个标签，并将其列设置为1，行设置为0

		# button被点击之后会被执行

		def begin():  # 当acction被点击时,该函数则生效

			action1.configure(text='已开启' + number.get() + '号摄像头检测')  # 设置button显示的内容


			action1.configure(state='disabled')  # 将按钮设置为灰色状态，不可使用状态

			num = number.get()

			module = hub.Module(name="pyramidbox_lite_server_mask", version='1.2.0')

			# result_path = './result'
			# if not os.path.exists(result_path):
			# 	os.mkdir(result_path)

			# name = "./result/1-mask_detection.mp4"
			# width = 1920
			# height = 1080
			# fps = 120
			# fourcc = cv2.VideoWriter_fourcc(*'vp90')
			# writer = cv2.VideoWriter(name, fourcc, fps, (width, height))

			maskIndex = 0
			index = 0
			data = []

			capture = cv2.VideoCapture(int(num))  # 打开摄像头
			# capture = cv2.VideoCapture('./test_video.mp4')  # 打开视频文件
			while True:
				frameData = {}
				ret, frame = capture.read()  # frame即视频的一帧数据
				if ret == False:
					break

				frame_copy = frame.copy()
				input_dict = {"data": [frame]}
				results = module.face_detection(data=input_dict, shrink=1, use_multi_scale=True)

				maskFrameDatas = []
				for result in results:
					label = result['data']['label']
					confidence_origin = result['data']['confidence']
					confidence = round(confidence_origin, 2)
					confidence_desc = str(confidence)

					top, right, bottom, left = int(result['data']['top']), int(
						result['data']['right']), int(result['data']['bottom']), int(
						result['data']['left'])

					# #将当前帧保存为图片
					img_name = "avatar_%d.png" % (maskIndex)
					# path = "./result/" + img_name
					# image = frame[top - 10:bottom + 10, left - 10:right + 10]
					# cv2.imwrite(path, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

					maskFrameData = {}
					maskFrameData['top'] = top
					maskFrameData['right'] = right
					maskFrameData['bottom'] = bottom
					maskFrameData['left'] = left
					maskFrameData['confidence'] = float(confidence_origin)
					maskFrameData['label'] = label
					maskFrameData['img'] = img_name

					maskFrameDatas.append(maskFrameData)

					maskIndex += 1

					color = (0, 255, 0)
					label_cn = "有口罩"
					if label == 'NO MASK':
						color = (0, 0, 255)
						label_cn = "无口罩"

					cv2.rectangle(frame_copy, (left, top), (right, bottom), color, 3)
					cv2.putText(frame_copy, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
				# origin_point = (left, top - 36)
				# frame_copy = paint_chinese(frame_copy, label_cn, origin_point, 24,
				#                           color)

				# writer.write(frame_copy)

				cv2.imshow('Mask Detection', frame_copy)

				frameData['frame'] = index
				# frameData['seconds'] = int(index/fps)
				frameData['data'] = maskFrameDatas

				data.append(frameData)
				# print(json.dumps(frameData)) # 输出检测结果

				index += 1

				cv2.waitKey(1)

			cv2.destroyAllWindows()

		# opencv输出中文
		def paint_chinese(im, chinese, position, fontsize, color_bgr):
			# 图像从OpenCV格式转换成PIL格式
			img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
			font = ImageFont.truetype(
				'SourceHanSansSC-Medium.otf', fontsize, encoding="utf-8")
			# color = (255,0,0) # 字体颜色
			# position = (100,100)# 文字输出位置
			color = color_bgr[::-1]
			draw = ImageDraw.Draw(img_PIL)
			# PIL图片上打印汉字 # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
			draw.text(position, chinese, font=font, fill=color)
			img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)  # PIL图片转cv2 图片
			return img

		def callbackClose():
			tkinter.messagebox.showwarning(title='警告', message='点击了关闭按钮')
			sys.exit(0)



		# 按钮
		win.protocol("WM_DELETE_WINDOW", callbackClose)

		action1 = ttk.Button(win, text="开始", command=begin)  # 创建一个按钮, text：显示按钮上面显示的文字, command：当这个按钮被点击之后会调用command函数

		action1.grid(column=2, row=1)  # 设置其在界面中出现的位置 column代表列 row 代表行



		# 创建一个下拉列表

		number = tk.StringVar()

		numberChosen = ttk.Combobox(win, width=12, textvariable=number)

		numberChosen['values'] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)  # 设置下拉列表的值

		numberChosen.grid(column=0, row=1)  # 设置其在界面中出现的位置 column代表列 row 代表行

		numberChosen.current(0)  # 设置下拉列表默认显示的值，0为 numberChosen['values'] 的下标值

		win.mainloop()  # 当调用mainloop()时,窗口才会显示出来

	def m_button4OnButtonClick(self, event):
		event.Skip()
		sys.exit(0)






app = wx.App()

main_win = MyFrame1(None)
main_win.Show()

app.MainLoop()


