import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

def find_matching_keypoints(image1, image2):
    #Input: two images (numpy arrays)
    #Output: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    sift = cv2.SIFT_create() #Scale-Invariant Feature Transform
    kp1, desc1 = sift.detectAndCompute(image1, None)
    kp2, desc2 = sift.detectAndCompute(image2, None)

    FLANN_INDEX_KDTREE = 0 #Fast Library for Approximate Nearest Neighbors, using k-dimensional tree
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good = []
    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2

def drawlines(img1,img2,lines,pts1,pts2):
    #img1: image on which we draw the epilines for the points in img2
    #lines: corresponding epilines
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def FindFundamentalMatrix(pts1, pts2):
    #Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    #Output: fundamental matrix (numpy array of shape (3, 3))

    # Assert that there are at least 8 points
    assert pts1.shape[0] >= 8 and pts2.shape[0] >= 8, "At least 8 points are required."

    # Normalize the points
    def normalize_points(points):
        # Compute the centroid of the points
        centroid = np.mean(points, axis=0)

        # Translate points to have centroid at the origin
        translated_points = points - centroid

        # Compute the average distance to the origin
        avg_dist = np.mean(np.sqrt(np.sum(translated_points ** 2, axis=1)))

        # Scale factor to make average distance sqrt(2)
        scale = np.sqrt(2) / avg_dist

        # Construct the transformation matrix
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])

        # Apply the transformation
        normalized_points = np.dot(T, np.vstack((points.T, np.ones(points.shape[0]))))


        return normalized_points[:2].T, T

    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    # Form the matrix A
    A = np.zeros((pts1_norm.shape[0], 9))
    for i in range(pts1_norm.shape[0]):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]

    # Compute the singular value decomposition of A
    U, S, Vt = np.linalg.svd(A)

    # The fundamental matrix is the last column of V (or the last row of V transpose)
    F = Vt[-1].reshape(3, 3)

    # Enforce the rank-2 constraint
    Uf, Sf, Vtf = np.linalg.svd(F)
    Sf[2] = 0  # Set the smallest singular value to zero
    F = np.dot(Uf, np.dot(np.diag(Sf), Vtf))

    F = np.dot(T2.T, np.dot(F, T1))
    F = F * (1 / F[2, 2])
    return F

def FindFundamentalMatrixRansac(pts1, pts2, num_trials = 1000, threshold = 1.0):
    #Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    #Output: fundamental matrix (numpy array of shape (3, 3))

    maxInliers = []
    bestF = None

    for _ in range(num_trials):
        # Randomly select 8 points
        indices = np.random.choice(len(pts1), 8, replace=False)
        sample_pts1 = pts1[indices]
        sample_pts2 = pts2[indices]

        # Compute the fundamental matrix for these 8 points
        F = FindFundamentalMatrix(sample_pts1, sample_pts2)

        # Calculate the number of inliers
        inliers = []
        for i in range(len(pts1)):
            pt1 = np.append(pts1[i], 1)
            pt2 = np.append(pts2[i], 1)

            # Compute the error
            error = np.dot(pt2.T, np.dot(F, pt1))
            if abs(error) < threshold:
                inliers.append(i)

        # Update the best model if the current one has more inliers
        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            bestF = F

    return bestF

def index_to_label(index):
    # Converts a zero-based index to a spreadsheet-style label.
    label = ''
    while index >= 0:
        label = chr(97 + (index % 26)) + label
        index = index // 26 - 1
    return label

def resize_to_same_height(img1, img2):
    target_height = max(img1.shape[0], img2.shape[0])

    scale_img1 = target_height / img1.shape[0]
    scale_img2 = target_height / img2.shape[0]

    resized_img1 = cv2.resize(img1, None, fx=scale_img1, fy=scale_img1, interpolation=cv2.INTER_AREA)
    resized_img2 = cv2.resize(img2, None, fx=scale_img2, fy=scale_img2, interpolation=cv2.INTER_AREA)

    return resized_img1, resized_img2

if __name__ == '__main__':
    #Set parameters
    data_path = './data'
    output_path = './output'
    use_ransac = False
    dpi = 200

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image_pairs = [('myleft.jpg', 'myright.jpg'), ('notredam_1.jpg', 'notredam_2.jpg'), ('mount_rushmore_1.jpg', 'mount_rushmore_2.jpg')]
    fig, axes = plt.subplots(len(image_pairs) * 2, 2, figsize=(15, len(image_pairs) * 5))  # Adjust the size as needed

    for idx, (image1_name, image2_name) in enumerate(image_pairs):
        #Load images
        image1_path = os.path.join(data_path, image1_name)
        image2_path = os.path.join(data_path, image2_name)
        image1 = np.array(Image.open(image1_path).convert('L'))
        image2 = np.array(Image.open(image2_path).convert('L'))


        #Find matching keypoints
        pts1, pts2 = find_matching_keypoints(image1, image2)

        #Builtin opencv function for comparison
        F_true = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]

        #todo: FindFundamentalMatrix
        if use_ransac:
            F = FindFundamentalMatrixRansac(pts1, pts2)
        else:
            F = FindFundamentalMatrix(pts1, pts2)
        print(f'`{image1_name}`-`{image2_name}`RANSAC_', use_ransac, F)
        print(f'`{image1_name}`-`{image2_name}`OpenCV', F_true)
        print(f'`{image1_name}`-`{image2_name}`mse', np.mean((F - F_true) ** 2))

        # Find epilines corresponding to points in second image,  and draw the lines on first image
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        img1, img2 = drawlines(image1, image2, lines1, pts1, pts2)
        img1, img2 = resize_to_same_height(img1, img2)
        combined_img = np.hstack((img1, img2))
        axes[idx * 2, 0].imshow(combined_img)
        axes[idx * 2, 0].set_title(f'({index_to_label(idx * 4)}) Custom for `{image1_name}`-`{image2_name}')
        axes[idx * 2, 0].axis('off')


        # Find epilines corresponding to points in first image, and draw the lines on second image
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        img1, img2 = drawlines(image2, image1, lines2, pts2, pts1)
        img1, img2 = resize_to_same_height(img1, img2)
        combined_img = np.hstack((img1, img2))
        axes[idx * 2 + 1, 0].imshow(combined_img)
        axes[idx * 2 + 1, 0].set_title(f'({index_to_label(idx * 4 + 2)}) Custom for `{image2_name}`-`{image1_name}`')
        axes[idx * 2 + 1, 0].axis('off')

        # Find epilines corresponding to points in second image,  and draw the lines on first image
        lines3 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F_true)
        lines3 = lines3.reshape(-1, 3)
        img1, img2 = drawlines(image1, image2, lines3, pts1, pts2)
        img1, img2 = resize_to_same_height(img1, img2)
        combined_img = np.hstack((img1, img2))
        axes[idx * 2, 1].imshow(combined_img)
        axes[idx * 2, 1].set_title(f'({index_to_label(idx * 4 + 1)}) OpenCV for `{image1_name}`-`{image2_name}`')
        axes[idx * 2, 1].axis('off')



        # Find epilines corresponding to points in first image, and draw the lines on second image
        lines4 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F_true)
        lines4 = lines4.reshape(-1, 3)
        img1, img2 = drawlines(image2, image1, lines4, pts2, pts1)
        img1, img2 = resize_to_same_height(img1, img2)
        combined_img = np.hstack((img1, img2))
        axes[idx * 2 + 1, 1].imshow(combined_img)
        axes[idx * 2 + 1, 1].set_title(f'({index_to_label(idx * 4 + 3)}) OpenCV for `{image2_name}`-`{image1_name}`')
        axes[idx * 2 + 1, 1].axis('off')

        # Save the large figure
    output_figure_path = os.path.join(output_path, f'{use_ransac}_all_result.jpg')
    plt.savefig(output_figure_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    plt.show()






