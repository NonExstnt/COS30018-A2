import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import subprocess

def run_main():
    lane = lane_var.get()
    scat = scat_var.get()

    command = f"python main.py --lane {lane} --scat {scat}"
    
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def run_training():
    model = model_var.get()
    lane = lane_var.get()
    scat = scat_var.get()
    aes = aes_var.get()
    hidden_sizes = hidden_sizes_var.get()
  
    
    if model == "all":
        for i in models:
            if i == "all":
                continue
            else:
                command = f"python train.py --model {i} --lane {lane} --scat {scat} --aes {aes} --hiddenSizes {hidden_sizes}"
                try:
                    subprocess.run(command, check=True, shell=True)
                except subprocess.CalledProcessError as e:
                    messagebox.showerror("Error", f"An error occurred: {e}")
                    return
        messagebox.showinfo("Success", "Training completed successfully!")
    else:
        command = f"python train.py --model {model} --lane {lane} --scat {scat} --aes {aes} --hiddenSizes {hidden_sizes}"
        try:
            subprocess.run(command, check=True, shell=True)
            messagebox.showinfo("Success", "Training completed successfully!")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

def toggle_saes_fields():
    if ((model_var.get() == "saes") or (model_var.get() == "all")):
        aes_entry.config(state="normal")
        hidden_sizes_entry.config(state="normal")
    else:
        aes_entry.config(state="disabled")
        hidden_sizes_entry.config(state="disabled")

def toggle_selector_fields():
    if selector_var.get() == "Train":
        training_button.grid(columnspan=2, row=8, padx=10, pady=20)
        predict_button.grid_forget()
        for i in model_radios:
            i.config(state="normal")
    else:
        predict_button.grid(columnspan=2, row=8, padx=10, pady=20)
        training_button.grid_forget()
        for i in model_radios:
            i.config(state="disabled")

# Create the main window
root = tk.Tk()
root.title("Train Model GUI")

# Create and place the widgets
ttk.Label(root, text="Select:").grid(column=0, row=0, padx=10, pady=5)
selector_var = tk.StringVar(value="Train")

selectors = ["Train", "Predict"]
for i, selector in enumerate(selectors):
    ttk.Radiobutton(root, text=selector, variable=selector_var, value=selector, command=toggle_selector_fields).grid(column=(i+1), row=0, padx=10, pady=5)

ttk.Label(root, text="Model:").grid(column=0, row=2, padx=10, pady=5)
model_var = tk.StringVar(value="lstm")

models = ["lstm", "gru", "saes", "rnn", "all"]
model_radios = []
for i, model in enumerate(models):
    rb = ttk.Radiobutton(root, text=model.upper(), variable=model_var, value=model, command=toggle_saes_fields)
    rb.grid(column=1, row=(i+2), padx=10, pady=5)
    model_radios.append(rb)

# Add a visible divider between columns 1 & 2
ttk.Separator(root, orient='vertical').grid(column=2, row=2, rowspan=6, sticky='ns', padx=10)
ttk.Separator(root, orient='horizontal').grid(column=0, columnspan=5, row=1, sticky='ew', pady=5)

ttk.Label(root, text="Lane:").grid(column=3, row=2, padx=10, pady=5)
lane_var = tk.IntVar(value=1)
ttk.Entry(root, textvariable=lane_var).grid(column=4, row=2, padx=10, pady=5)

ttk.Label(root, text="SCAT:").grid(column=3, row=3, padx=10, pady=5)
scat_var = tk.IntVar(value=970)
ttk.Entry(root, textvariable=scat_var).grid(column=4, row=3, padx=10, pady=5)

ttk.Label(root, text="AES:").grid(column=3, row=4, padx=10, pady=5)
aes_var = tk.IntVar(value=3)
aes_entry = ttk.Entry(root, textvariable=aes_var)
aes_entry.grid(column=4, row=4, padx=10, pady=5)

ttk.Label(root, text="Hidden Sizes:").grid(column=3, row=5, padx=10, pady=5)
hidden_sizes_var = tk.StringVar(value="32,32")
hidden_sizes_entry = ttk.Entry(root, textvariable=hidden_sizes_var)
hidden_sizes_entry.grid(column=4, row=5, padx=10, pady=5)

toggle_saes_fields()  # Initialize the state of AES and Hidden Sizes fields

training_button = ttk.Button(root, text="Run Training", command=run_training)
predict_button = ttk.Button(root, text="Run Predict", command=run_main)

toggle_selector_fields()

# Run the application
root.mainloop()
