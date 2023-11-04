import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2
import pytesseract

# create the sudoku grid
grid = np.array([[0,2,0,5,0,1,0,9,0],
                  [8,0,0,2,0,3,0,0,6],
                  [0,3,0,0,6,0,0,7,0],
                  [0,0,1,0,0,0,6,0,0],
                  [5,4,0,0,0,0,0,1,9],
                  [0,0,2,0,0,0,7,0,0],
                  [0,9,0,0,3,0,0,8,0],
                  [2,0,0,8,0,4,0,0,7],
                  [0,1,0,9,0,7,0,6,0]])



# isSafe function where it checks the horizontal, vertical, and 3x3 boxes for sudoku
def isSafe(row, col, num):
    global grid
    for x in range(9):
        if grid[row][x] == num:
            return False
    for y in range(9):
        if grid[y][col] == num:
            return False
    x0 = row-row % 3
    y0 = col-col % 3
    for i in range(3):
        for j in range(3):
            if grid[i+x0][j+y0] == num:
                return False
    return True

counter = -1
# gui part
class SudokuApp(tk.Tk):

    def __init__(self):
        global grid
        super().__init__()
        self.title('Sudoku App')
        self.geometry("400x400")
        cnt = 0
        for x in range(9):
            for y in range(9):
                if grid[x][y] != 0:
                    # need dynamic variables and call them dynamically
                    globals()['label%s' % cnt] = ttk.Label(self, borderwidth=1, text="{}".format(grid[x][y]))
                    globals()['label%s' % cnt].grid(row=x, column=y)
                else:
                    globals()['label%s' % cnt] = ttk.Label(self, text="", borderwidth=1)
                    globals()['label%s' % cnt].grid(row=x, column=y)
                cnt += 1
        self.backtracking()

    # backtracking algorithm
    def backtracking(self):
        global grid, counter
        for y in range(9):
            for x in range(9):
                counter += 1
                if grid[y][x] == 0:
                    for n in range(1, 10):
                        if isSafe(y, x, n):
                            grid[y][x] = n
                            # shows the changes done to the sudoku grid
                            globals()['label%s' % counter].configure(text="{}".format(n))
                            # globals()['label%s' % cnt].grid(row=y, column=x)
                            self.backtracking()
                            """
                            the problem with calling dynamic variables in this recursive function is that I can't 
                            properly count it like in the init function
                            """
                            counter -= 1
                            grid[y][x] = 0
                            # visualizes the algorithm
                            globals()['label%s' % counter].configure(text="")
                            # globals()['label%s' % cnt].grid(row=y, column=x)
                    return
        print(grid)


App = SudokuApp()
App.mainloop()

# opencv part


# input from an image part