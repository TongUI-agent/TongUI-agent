
from PIL import Image, ImageDraw
from qwen_vl_utils import smart_resize

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

def resize_image(original_image, factor=28, min_pixels=336*336, max_pixels=768*768):
    """
    Resize the image using smart_resize function to meet the model's requirements.
    
    Args:
        original_image: PIL Image object
        factor: The factor by which dimensions should be divisible (default: 14)
        min_pixels: Minimum total pixels (default: 336*336)
        max_pixels: Maximum total pixels (default: 672*672)
    
    Returns:
        Resized PIL Image
    """
    # Get original dimensions
    original_width, original_height = original_image.size
    
    # Calculate new dimensions using smart_resize
    new_width, new_height = smart_resize(
        height=original_height,
        width=original_width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
    
    # Resize the image while maintaining aspect ratio
    resized_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    print("Original image: ", original_image.size)
    print("Resized image: ", resized_image.size)
    return resized_image

if __name__ == "__main__":
    # Example usage
    image_path = "assets/safari_google.png"
    point = [0.47, 0.51]  # Example coordinates from inference output
    draw_point_on_image(image_path, point)
