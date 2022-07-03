from PIL import Image, ImageDraw, ImageFont


class ImageWithTextGenerator:
    def __init__(self, img_width, img_height, font_path, font_size):
        self.img_width = img_width
        self.img_height = img_height
        self.font = ImageFont.truetype(font_path, font_size)

    def generate_image(self, text, filepath):
        image = Image.new(mode="RGB", size=(self.img_width, self.img_height))
        draw = ImageDraw.Draw(image)

        text_width, text_height = draw.textsize(text, self.font)
        x = (self.img_width - text_width) / 2
        y = (self.img_height - text_height) / 2
        draw.text((x, y), text, font=self.font, fill="#FFFFFF")

        image.save(filepath)
