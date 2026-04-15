import cv2
import numpy as np

INPUT_PATH = 'image (6).png'
FINAL_SIZE = 200
PIXEL_SIZE = 80
NUM_COLORS = 4
BG_COLOR = [255, 255, 255]
image = cv2.imread(INPUT_PATH)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
edges = cv2.dilate(edges, None, iterations=1)
cv2.imwrite('1_edges.png', edges)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(gray)
if contours:
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
cv2.imwrite('2_mask.png', mask)

background = np.full_like(image, BG_COLOR)
flattened = np.where(mask[:, :, None] == 255, image, background)
cv2.imwrite('3_flattened.png', flattened)

cv2.imshow('Edges', edges)
cv2.imshow('Mask', mask)
cv2.imshow('Flattened', flattened)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Done! Outputs saved:")
print("1_edges.png")
print("2_mask.png")
print("3_flattened.png")
