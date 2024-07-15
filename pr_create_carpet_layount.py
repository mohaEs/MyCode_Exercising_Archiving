import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
from sklearn.cluster import KMeans
import numpy as np
from matplotlib.patches import Patch

def resize_image_with_aspect_ratio(image, target_size):
    # Get original dimensions
    original_height, original_width = image.shape[:2]
    
    # Calculate the target dimensions while maintaining the aspect ratio
    aspect_ratio = original_width / original_height
    if aspect_ratio > 1:
        # Landscape orientation
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        # Portrait orientation
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
    
    # Resize the image
    resized_image = resize(image, (new_height, new_width), anti_aliasing=True)
    return resized_image





# Load the image
image_path = "C:\\Users\\Moha-Cate\\Downloads\\yaserina.jpg"  # Replace with your image path
image = io.imread(image_path)

# Resize the image
# Resize the image while maintaining the aspect ratio
############ 10 clots in each centimeter
############ height 56 centimeter
target_size = 560  # The larger dimension will be resized to 256 pixels
image_resized = resize_image_with_aspect_ratio(image, target_size)

# Reshape the image to a 2D array of pixels
pixels = image_resized.reshape(-1, 3)

# Quantize the colors using KMeans
quantize_colors = 10  # Number of colors to quantize to
kmeans = KMeans(n_clusters=quantize_colors, random_state=42).fit(pixels)
labels = kmeans.predict(pixels)
quantized_image = kmeans.cluster_centers_[labels].reshape(image_resized.shape)#.astype(np.uint8)

# Create a legend for the quantized colors
unique_colors = kmeans.cluster_centers_

if np.max(quantized_image) <=1:
    quantized_image = (255*quantized_image).astype(np.uint8)
    unique_colors = (255*kmeans.cluster_centers_).astype(np.uint8)
else:
    quantized_image =  quantized_image.astype(np.uint8)
    unique_colors = kmeans.cluster_centers_.astype(np.uint8)

legend_patches = [Patch(color=np.array(color)/255, label=f'RGB: {tuple(color)}') for color in unique_colors]
    
# Show the original and processed images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Original image
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis('off')

# Processed image
ax[1].imshow(quantized_image)
ax[1].set_title(f"Resized and Quantized Image ({quantize_colors} colors)")
ax[1].axis('off')

# Add legend to the plot
# plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.5, 1))
plt.show()


# Show the legend in a separate figure
fig_legend = plt.figure(figsize=(6, 6))
plt.legend(handles=legend_patches, loc='center', frameon=False)
plt.axis('off')
plt.show()