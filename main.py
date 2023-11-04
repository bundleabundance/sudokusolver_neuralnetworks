import PIL.Image
import numpy as np
import cv2
import pytesseract
import pygame
import tensorflow as tf

efficientdet_model = tf.keras.applications.EfficientNetB0(weights='imagenet')
image_path = 'sudoku.jpg'


# create the sudoku grid
grid = np.array([[0, 2, 0, 5, 0, 1, 0, 9, 0],
                 [8, 0, 0, 2, 0, 3, 0, 0, 6],
                 [0, 3, 0, 0, 6, 0, 0, 7, 0],
                 [0, 0, 1, 0, 0, 0, 6, 0, 0],
                 [5, 4, 0, 0, 0, 0, 0, 1, 9],
                 [0, 0, 2, 0, 0, 0, 7, 0, 0],
                 [0, 9, 0, 0, 3, 0, 0, 8, 0],
                 [2, 0, 0, 8, 0, 4, 0, 0, 7],
                 [0, 1, 0, 9, 0, 7, 0, 6, 0]])


# isSafe function where it checks the horizontal, vertical, and 3x3 boxes for sudoku
def is_safe(row, col, num):
    global grid
    for x in range(9):
        if grid[row][x] == num:
            return False
    for y in range(9):
        if grid[y][col] == num:
            return False
    x0 = row - row % 3
    y0 = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + x0][j + y0] == num:
                return False
    return True


# some constants needed for the gui
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
HEIGHT, WIDTH = 450, 450
pygame.font.init()
number_font = pygame.font.SysFont('Helvetica', 25)
FPS = 100
clock = pygame.time.Clock()
solved = False


# gui part
class SudokuApp:
    def __init__(self):
        global grid
        # initialize the game window and give the basic features
        pygame.init()
        self.WIN = pygame.display.set_mode((WIDTH, HEIGHT))
        self.WIN.fill(BLACK)
        pygame.display.set_caption('SudokuApp')
        running = True

        # draw the initial grid
        self.draw_grid()

        # initialize the main loop of the window
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.backtracking()
            pygame.display.update()
        pygame.quit()

    # this can be done in main loop as well, but it would constantly redraw the grid
    def draw_grid(self):
        global grid
        # draw the needed grid
        block_size = 50
        for x in range(9):
            for y in range(9):
                # need rectangles for writing down the numbers
                rect = pygame.Rect(x * block_size, y * block_size, block_size, block_size)
                pygame.draw.rect(self.WIN, WHITE, rect, 1)
                # get the filled values from the grid
                if grid[y][x] != 0:
                    number_image = number_font.render("{}".format(grid[y][x]), True, WHITE, BLACK)
                    self.WIN.blit(number_image, ((x * block_size + 17), (y * block_size + 17)))
                else:
                    pass
            # 3x3 borders are drawn different
            if x % 3 == 0 and x % 8 != 0:
                pygame.draw.line(self.WIN, WHITE, (x * block_size, 0), (x * block_size, HEIGHT), 7)
                pygame.draw.line(self.WIN, WHITE, (0, x * block_size), (WIDTH, x * block_size), 7)
            pygame.display.flip()

    def backtracking(self):
        block_size = 50
        # for slower visualization implemented time control
        clock.tick(FPS)
        global grid, solved
        for y in range(9):
            for x in range(9):
                rect = pygame.Rect(x * block_size + 3, y * block_size + 3, block_size - 5, block_size - 5)
                if grid[y][x] == 0:
                    for n in range(1, 10):
                        if is_safe(y, x, n):
                            grid[y][x] = n
                            # shows the changes done to the sudoku grid
                            number_image = number_font.render("{}".format(grid[y][x]), True, WHITE, BLACK)
                            self.WIN.blit(number_image, ((x * block_size + 17), (y * block_size + 17)))
                            pygame.display.update()
                            self.backtracking()
                            # for slower visualization implemented time control
                            clock.tick(FPS)
                            # this algorithm will find all solutions
                            # for the first solution need a boolean variable
                            if solved:
                                return
                            grid[y][x] = 0
                            # visualizes the algorithm
                            # erases the wrong guess and when other guesses are made and the box is cleared
                            self.WIN.fill(BLACK, rect)
                            pygame.display.update()
                    return
        solved = True


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  # Resize the image to match the input size of the model
    image = image / 255.0  # Normalize the image pixel values to the range [0, 1]
    return image

def extract_digits(image_path, model):
    preprocessed_image = preprocess_image(image_path)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add a batch dimension

    # Use the EfficientDet model to predict bounding boxes of digits
    predictions = model.predict(preprocessed_image)
    print(predictions)
    # Initialize lists to store the extracted digits and their bounding boxes
    digits = []
    digit_bboxes = []

    # Process the predictions to extract individual digits and their bounding boxes
    for prediction in predictions:
        xmin = prediction[0]
        ymin = prediction[1]
        xmax = prediction[2]
        ymax = prediction[3]

        # Extract the digit image using the bounding box coordinates
        digit_image = preprocessed_image[0, int(ymin):int(ymax), int(xmin):int(xmax)]

        # Classify the digit using a digit classifier (Using pytesseract as a placeholder)
        digit = int(pytesseract.image_to_string(digit_image, config='--psm 6').strip())

        # Associate the classified digit with its bounding box
        digits.append(digit)
        digit_bboxes.append((xmin, ymin, xmax, ymax))
    print(digits, digit_bboxes)
    return digits, digit_bboxes



def classify_digits(extracted_digits):
    # Use a pre-trained digit classification model (e.g., MNIST classifier) to classify the extracted digits
    # You can use TensorFlow's MNIST classifier or any other pre-trained model for this task

    # For now, we will assume all digits are correctly recognized and classified

    return extracted_digits

# Function to reconstruct the Sudoku grid
def reconstruct_grid(classified_digits):
    sudoku_grid = [[0 for _ in range(9)] for _ in range(9)]
    for digit_info in classified_digits:
        row, col, digit = digit_info
        sudoku_grid[row][col] = digit
    return sudoku_grid

# run the main program
if __name__ == "__main__":
    extract_digits('sudoku.jpg', efficientdet_model)
    #SudokuApp()
