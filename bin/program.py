import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define upper and lower control limits (you can adjust these values as needed)
upper_control_limit = 70  # Example upper limit
lower_control_limit = 60  # Example lower limit

# Number of data points to consider for control chart (X-bar chart)
n_points = 5

def calculate_fill_percentage(aspect_ratio):
    # Assuming aspect ratio represents the fill percentage linearly
    return (1 - aspect_ratio) * 100

def check_control_limits(fill_percentage):
    if fill_percentage > upper_control_limit:
        return "Above Upper Limit"
    elif fill_percentage < lower_control_limit:
        return "Below Lower Limit"
    else:
        return "Within Control Limits"

def calculate_mean_std(fill_percentages):
    mean = np.mean(fill_percentages)
    std_dev = np.std(fill_percentages)
    return mean, std_dev

def check_control_chart(fill_percentages):
    mean, std_dev = calculate_mean_std(fill_percentages)
    upper_limit = mean + (3 * std_dev)
    lower_limit = mean - (3 * std_dev)
    return upper_limit, lower_limit

# List of image paths
image_paths = ["./TestImages/50ml.jpg", "./TestImages/75ml.jpg", "./TestImages/100ml.jpg", "./TestImages/150ml.jpg", "./TestImages/200ml.jpg" ]

# Initialize lists to store fill percentages and confidence levels for control chart
fill_percentages = []
confidence_levels = []

# Iterate over each image path
for image_path in image_paths:
    bottle_3_channel = cv2.imread(image_path)

    if bottle_3_channel is None:
        print(f"Error: Unable to read the image file: {image_path}")
    else:
        img_gray = cv2.cvtColor(bottle_3_channel, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=600)

        (T, bottle_threshold) = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY_INV)

        contours = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        areas = [cv2.contourArea(contour) for contour in contours]
        (contours, areas) = zip(*sorted(zip(contours, areas), key=lambda a:a[1]))

        bottle_clone = bottle_3_channel.copy()
        (x, y, w, h) = cv2.boundingRect(contours[-1])
        aspectRatio = w / float(h)
        fill_percentage = calculate_fill_percentage(aspectRatio)
        print(f"Fill Percentage for {image_path}: {fill_percentage}")

        control_status = check_control_limits(fill_percentage)

        # Calculate z-score and confidence level
        z_score = (fill_percentage - np.mean(fill_percentages)) / np.std(fill_percentages)
        confidence_level = norm.cdf(z_score) * 100
        
        rate = 0.0012  # Random gas rate
        gas_fees = 0.005  # Random gas fees
        chain = "Ethereum"  # Random blockchain chain
        txn_time = "2024-03-11 14:30:00"  # Random transaction time
        party_address = "0x4Dc38d3D51F94d3e632a2d3eC37C21bC8B8C8E6D"  # Random party address

      
        bluff_text = f"Gas Rate: {rate}, Gas Fees: {gas_fees}, Chain: {chain}, Time: {txn_time}, Address: {party_address}"
        cv2.putText(bottle_clone, bluff_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

        # Add fill percentage and confidence level to the lists for control chart
        fill_percentages.append(fill_percentage)
        confidence_levels.append(confidence_level)

        # Add text on image
        text = f"{control_status} ({fill_percentage:.2f}%, {confidence_level:.2f}% Confidence)"
        cv2.putText(bottle_clone, text, (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

        # Draw rectangle on image
        if control_status == "Above Upper Limit":
            cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for above upper limit
        elif control_status == "Below Lower Limit":
            cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for below lower limit
        else:
            cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for within control limits

        cv2.imshow("Decision", bottle_clone)
        cv2.waitKey(0)

# Plot histogram of fill percentages
plt.figure(figsize=(8, 6))
plt.hist(fill_percentages, bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of Fill Percentages')
plt.xlabel('Fill Percentage')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Calculate mean and standard deviation for control chart
mean, std_dev = calculate_mean_std(fill_percentages)
print(f"Mean of fill percentages: {mean}")
print(f"Standard deviation of fill percentages: {std_dev}")

# Check control chart
upper_limit, lower_limit = check_control_chart(fill_percentages)
print(f"Upper control limit: {upper_limit}")
print(f"Lower control limit: {lower_limit}")

# Plot control chart
plt.figure(figsize=(10, 6))
plt.plot(fill_percentages, marker='o', color='blue', linestyle='-')
plt.axhline(upper_limit, color='red', linestyle='--', label='Upper Control Limit')
plt.axhline(lower_limit, color='green', linestyle='--', label='Lower Control Limit')
plt.title('Control Chart of Fill Percentages')
plt.xlabel('Sample Number')
plt.ylabel('Fill Percentage')
plt.legend()
plt.grid(True)
plt.show()

# Scatter Diagram
def scatter_diagram(x, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue')
    plt.title('Scatter Diagram')
    plt.xlabel('Sample Number')
    plt.ylabel('Fill Percentage')
    plt.grid(True)
    plt.show()

# Call scatter diagram function
scatter_diagram(range(1, len(fill_percentages)+1), fill_percentages)
