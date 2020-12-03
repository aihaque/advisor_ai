from PIL import Image, ExifTags


def main():
    img = Image.open(r'C:\Users\tjsan\Documents\GitHub\advisor_ai\code\jay_3.jpeg')
    exif_data = img.getexif()
    print(exif_data)

if __name__ == '__main__':
    main()