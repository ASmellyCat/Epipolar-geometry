# Epipolar-geometry
 Use epipolar geometry to find the structure of a 3D scene

## 1 Introduction
In computer vision, the fundamental matrix encapsulates the intrinsic projective geometry between two views of a scene. Its estimation is pivotal for applications such as stereo vision, motion tracking, and the extraction of 3D structure from image pairs. This report showcases a side-by-side evaluation of the fundamental matrix as computed by the traditional 8-point algorithm and a RANSAC-based approach, against the backdrop of results from OpenCV’s built-in methods. By contrasting these techniques, we aim to discern the comparative efficacy and robustness of the custom-implemented algorithms in relation to OpenCV’s established procedures.

## 2 Results
This section presents the results of the fundamental matrix estimation and the corre- sponding epipolar lines drawn on pairs of images. The analysis compares the results ob- tained from the custom implementation against those derived using the OpenCV built-in function. The evaluation focuses on both approaches’ performances with and without the RANSAC algorithm.

### 2.1 Without RANSAC
Figure 1 presents the epipolar lines computed without the RANSAC algorithm. Images (a), (c), (e), (g), (i), and (k) are the results of the custom implementation using the normalized 8-point algorithm. Images (b), (d), (f), (h), (j), and (l) are the results from the OpenCV built-in function.

The fundamental matrix F is a 3 × 3 matrix that enforces a constraint on the cor- responding points between two stereo images. Given a point x in the left image and its counterpart x′ in the right image, they satisfy the epipolar constraint x′⊤Fx = 0.

For each paired set of stereo images, the points in the left image have corresponding points the the same random color in the right image. The colorful lines in the left image represent the epipolar lines that correspond to the feature points in the right image. These lines are computed from the feature points in the right image and drawn onto the left image. We can see that the feature points in the left image align with these epipolar lines.

As shown in Table 1, the fundamental matrices derived from different image pairs are compared. The table demonstrates the consistency between my implementation and the OpenCV built-in function in terms of the epipolar geometry. The very small differences that shown in mse are caused by floating point computations.

![Figure 1: Comparison of epipolar lines estimated without RANSAC. Custom implemen- tation using the normalized 8-point algorithm (a, c, e, g, i, k). Results using the OpenCV built-in function (b, d, f, h, j, l).
](https://github.com/ASmellyCat/Epipolar-geometry/assets/110814688/dfc3d761-a72b-49c8-abf2-19508125a55a)

<img width="161" alt="Screen Shot 2023-12-21 at 19 34 25" src="https://github.com/ASmellyCat/Epipolar-geometry/assets/110814688/58af2c2d-0e8b-4963-92f7-ebcd35245df2">


### 2.2 With RANSAC
Figure 2 demonstrates the epipolar lines computed with the RANSAC algorithm. Images (a), (c), (e), (g), (i), and (k) show the results of the custom implementation incorporating RANSAC for outlier rejection. Images (b), (d), (f), (h), (j), and (l) show the results obtained using the OpenCV function with RANSAC.

The performance of the fundamental matrix estimation methods is quantitatively compared in Table 2. This table presents the results for both the self-implemented RANSAC-based method and the OpenCV function across various image pairs.

The interpretation of feature points and epipolar lines are similar with section 2.1.

From the Figure 2 and Table 2, we can see that the RANSAC algorithm’s result is slightly different with OpenCV method, but it still makes sense.

RANSAC is a robust algorithm used to estimate models in data with outliers. It randomly selects subsets of the data to build a model and checks how well it fits the rest of the data. In the final step, RANSAC selects the model that has the highest number of inliers. Its main advantages are its strong resistance to outliers and its ability to produce reliable models even with noisy data. Variability in results arises from its random selection process, making it slightly different in each run. This randomness, however, is crucial for its effectiveness in real-world scenarios where data is often imperfect.

![Figure 2: Comparison of epipolar lines estimated with RANSAC. Custom implementation using RANSAC (a, c, e, g, i, k). Results using the OpenCV function (b, d, f, h, j, l).](https://github.com/ASmellyCat/Epipolar-geometry/assets/110814688/50a93c4a-b703-4dfd-9350-e99dbf069ff4)

<img width="164" alt="Screen Shot 2023-12-21 at 19 35 40" src="https://github.com/ASmellyCat/Epipolar-geometry/assets/110814688/8d6750c4-f5d1-4e9b-976a-71bf072c9af6">

## Conclusion

The above discussions and visual results corroborate the accuracy of the fundamental ma- trices calculated using both the normalized 8-point algorithm and the RANSAC-enhanced method. The outputs are consistent with the theoretical principles of epipolar geometry and validate the effectiveness of these computational approaches.
