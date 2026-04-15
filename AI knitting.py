from PIL import Image

img6 = Image.open("image (6).png")
img6 = img6.resize((80, 80), Image.NEAREST) #Change the size of the image
img6 = img6.convert("P", palette = Image.ADAPTIVE, colors=4)
img6 = img6.convert("RGB")

def color_comparing(color1, color2, threshold=50):
    return sum(abs(a - b) for a, b in zip(color1, color2)) < threshold

background_color = img6.getpixel((0,0)) #Top left pixel
new_background = (255, 255, 255) #Change this to any color I want
pixels = img6.load()
for x in range(img6.width):
    for y in range(img6.height):
        if color_comparing(pixels[x, y], background_color):
            pixels[x, y] = new_background

img6.show()
