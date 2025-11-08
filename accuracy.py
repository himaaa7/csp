import cv2
from skimage.metrics import structural_similarity as ssim

# Load before and after images
before = cv2.imread("dataset/before.png")
after = cv2.imread("dataset/after.png")

# Resize to same dimensions
before = cv2.resize(before, (600, 400))
after = cv2.resize(after, (600, 400))

# Convert to grayscale
before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# Compute SSIM
score, diff = ssim(before_gray, after_gray, full=True)
diff = (diff * 255).astype("uint8")

# Invert difference so higher values = more change
diff_inv = cv2.bitwise_not(diff)

# Threshold the diff image
_, thresh = cv2.threshold(diff_inv, 30, 255, cv2.THRESH_BINARY)

# Calculate percentage of changed pixels
changed_pixels = cv2.countNonZero(thresh)
total_pixels = diff_inv.size
change_percentage = (changed_pixels / total_pixels) * 100

# Cleanup percentage is how much change we see
print(f"🧹 Cleanup Progress: {round(change_percentage, 2)}% area changed")
