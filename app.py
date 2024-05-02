import tkinter
import pickle
import numpy as np


class MyGUI():
    def __init__(self):
        self.main_window = tkinter.Tk()
        self.main_window.title('ML Diamond Price Predictor')

        # Create a frame
        self.top_frame = tkinter.Frame(self.main_window)
        self.top_frame.grid()

        # Carat Label and Entry Widget
        self.carat_label = tkinter.Label(self.top_frame, text='Carat')
        self.carat_entry = tkinter.Entry(self.top_frame)

        self.carat_label.grid(row=0, column=0)
        self.carat_entry.grid(row=0, column=1)

        #X Label and Entry Widget
        self.x_label = tkinter.Label(self.top_frame, text='X')
        self.x_entry = tkinter.Entry(self.top_frame)

        self.x_label.grid(row=1, column=0)
        self.x_entry.grid(row=1, column=1)

        #Y Label and Entry Widget
        self.y_label = tkinter.Label(self.top_frame, text='Y')
        self.y_entry = tkinter.Entry(self.top_frame)

        self.y_label.grid(row=2, column=0)
        self.y_entry.grid(row=2, column=1)

        #Z Label and Entry Widget
        self.z_label = tkinter.Label(self.top_frame, text='Z')
        self.z_entry = tkinter.Entry(self.top_frame)

        self.z_label.grid(row=3, column=0)
        self.z_entry.grid(row=3, column=1)

        #Cut Label and Entry Widget
        self.cut_label = tkinter.Label(self.top_frame, text='Cut (Fair/Good/Very Good/Premium/Ideal)')
        self.cut_entry = tkinter.Entry(self.top_frame)

        self.cut_label.grid(row=4, column=0)
        self.cut_entry.grid(row=4, column=1)

        #Color Label and Entry Widget
        self.color_label = tkinter.Label(self.top_frame, text='Color (J/I/H/G/F/E/D)')
        self.color_entry = tkinter.Entry(self.top_frame)

        self.color_label.grid(row=5, column=0)
        self.color_entry.grid(row=5, column=1)

        #Clarity Label and Entry Widget
        self.clarity_label = tkinter.Label(self.top_frame, text='Clarity (I1/SI2/SI1/VS2/VS1/VVS2/VVS1/IF)')
        self.clarity_entry = tkinter.Entry(self.top_frame)

        self.clarity_label.grid(row=6, column=0)
        self.clarity_entry.grid(row=6, column=1)

        #Predict Button
        self.predict_button = tkinter.Button(self.top_frame,text = 'Predict', command = self.predict_price)
        self.predict_button.grid(row=7,column=0)

        #Entry Result
        self.prediction_var = tkinter.StringVar()
        self.output = tkinter.Entry(self.top_frame, state='readonly', textvariable=self.prediction_var)
        self.output.grid(row=8, column=0)

    #Function
    def predict_price(self): #Bind to Button
        cut_dictionary = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
        color_dictionary = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
        clarity_dictionary = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

        carat = float(self.carat_entry.get())
        x = float(self.x_entry.get())
        y = float(self.y_entry.get())
        z = float(self.z_entry.get())
        color = str(self.color_entry.get())
        cut = str(self.cut_entry.get())
        clarity = str(self.clarity_entry.get())

        with open('XGB_Model.pkl', 'rb') as file:
            LREG_Model = pickle.load(file)
         # Make prediction
        predicted_price = LREG_Model.predict([[carat, x, y, z, cut_dictionary[cut], color_dictionary[color], clarity_dictionary[clarity]]])
        displayed_price = np.exp(predicted_price[0])

        self.prediction_var.set(f'${displayed_price:.5f}')

my_gui = MyGUI()
tkinter.mainloop()
