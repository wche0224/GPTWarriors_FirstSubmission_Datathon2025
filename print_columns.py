import pandas as pd
import tkinter as tk
from tkinter import ttk


# Function to show data in a Tkinter window
def show_data_in_window(df):
    # Create a root window
    root = tk.Tk()
    root.title("Data Display")

    # Create a Treeview widget (for displaying tables)
    tree = ttk.Treeview(root, columns=list(df.columns), show="headings")
    
    # Define columns (headings)
    for column in df.columns:
        tree.heading(column, text=column)
        tree.column(column, width=200, anchor='center')  # Dynamically set column width

    # Insert data into table
    for row in df.itertuples(index=False):
        tree.insert('', 'end', values=row)

    # Scrollbar for the Treeview
    scrollbar = tk.Scrollbar(root, orient='vertical', command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side='right', fill='y')

    # Pack Treeview widget
    tree.pack(expand=True, fill='both')

    # Start the Tkinter main loop
    root.mainloop()
