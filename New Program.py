import cv2

image = cv2.imread('image (6).png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

edges = cv2.dilate(edges, None, iterations=1)

cv2.imwrite('knitting_pattern_outline.png', edges)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()