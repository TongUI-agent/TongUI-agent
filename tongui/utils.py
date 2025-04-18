
from PIL import Image, ImageDraw


def draw_point_on_image(image_path, point, output_path=None):
    """
    Draw a point on the image at the given coordinates (0-1 scale)

    Args:
        image_path: Path to the input image
        point: Tuple of (x, y) coordinates in 0-1 scale
        output_path: Path to save the output image. If None, will save as 'output.png'
    """
    # Open the image
    img = Image.open(image_path)
    width, height = img.size

    # Convert 0-1 coordinates to pixel coordinates
    x = int(point[0] * width)
    y = int(point[1] * height)

    # Create a drawing context
    draw = ImageDraw.Draw(img)

    # Draw a red circle at the point
    radius = 10
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="red")

    # Save the image
    if output_path is None:
        output_path = "output.png"
    img.save(output_path)
    print(f"Image saved to {output_path}")
    img.show()


if __name__ == "__main__":
    # Example usage
    image_path = "assets/safari_google.png"
    point = [0.47, 0.51]  # Example coordinates from inference output
    draw_point_on_image(image_path, point)
