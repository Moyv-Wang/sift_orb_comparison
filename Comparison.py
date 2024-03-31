import cv2
import numpy as np
import os
import random
def register_images_sift(img1_path, img2_path, matrix):
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Create FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find matches
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    print("SIFT Origin matches: ", len(good_matches))
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("Origin_Matches", img_matches)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Apply transformation matrix
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
    # 添加一列全为 1 的值，以表示齐次坐标
    src_pts_homogeneous = np.hstack((src_pts.squeeze(), np.ones((len(src_pts), 1))))
    # 扩展矩阵维度，以便进行矩阵乘法
    matrix = np.vstack((matrix, [0, 0, 1]))
    # 计算变换后的点坐标
    transformed_src_pts = np.dot(matrix, src_pts_homogeneous.T)
    # 将齐次坐标转换为二维坐标
    transformed_src_pts = transformed_src_pts[:2, :].T

    good_sum = 0
    new_src = []
    new_dst = []
    for i in range(len(transformed_src_pts)):
        dist = np.linalg.norm(transformed_src_pts[i] - dst_pts[i])
        # 如果距离小于 10，认为是正确的匹配
        if dist < 10:
            if transformed_src_pts[i][0] < 0 or transformed_src_pts[i][1] < 0:
                continue
            if transformed_src_pts[i][0] > img1.shape[1] or transformed_src_pts[i][1] > img1.shape[0]:
                continue
            good_sum += 1
            new_src.append(src_pts[i])
            new_dst.append(dst_pts[i])
            # new_good_matches.append(good_matches[i])
    src_pts = np.array(new_src)
    dst_pts = np.array(new_dst)
    # # 使用 RANSAC 算法估计单应性矩阵
    # M, mask = cv2.findHomography(np.array(new_src), np.array(new_dst), cv2.RANSAC, 5.0)
    # src_pts = []
    # dst_pts = []
    # for i in range(len(mask)):
    #     if mask[i] == 1:
    #         src_pts.append(new_src[i])
    #         dst_pts.append(new_dst[i])
    #绘制实际上的匹配点
    # 创建一个新的图像，将两幅输入图像拼接到一起
    new_img = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), np.uint8)
    new_img[:img1.shape[0], :img1.shape[1]] = img1
    #
    new_img[int((new_img.shape[0]-img2.shape[0])/2):int((new_img.shape[0]+img2.shape[0])/2), img1.shape[1]:] = img2
    # 使用cv2.circle()在图像上绘制点, cv2.line()绘制线
    for i in range(len(src_pts)):
        R = random.randint(0, 255)
        G = random.randint(0, 255)
        B = random.randint(0, 255)
        cv2.circle(new_img, (int(src_pts[i][0][0]), int(src_pts[i][0][1])), 2, (R, G, B), 2)
        cv2.circle(new_img, (int(dst_pts[i][0][0]) + img1.shape[1], int((dst_pts[i][0][1]) + (new_img.shape[0]-img2.shape[0])/2)), 2, (R, G, B), 2)
        cv2.line(new_img, (int(src_pts[i][0][0]), int(src_pts[i][0][1])), (int(dst_pts[i][0][0]) + img1.shape[1], int((dst_pts[i][0][1]) + (new_img.shape[0]-img2.shape[0])/2)), (R, G, B), 1)
        # cv2.circle(new_img, (int(src_pts[i][0][0]), int(src_pts[i][0][1])), 2, (R, G, B), 2)
        # cv2.circle(new_img, (int(transformed_src_pts[i][0]) + img1.shape[1], int((transformed_src_pts[i][1]) + (new_img.shape[0]-img2.shape[0])/2)), 2, (R, G, B), 2)
        # cv2.line(new_img, (int(src_pts[i][0][0]), int(src_pts[i][0][1])), (int(transformed_src_pts[i][0]) + img1.shape[1], int((transformed_src_pts[i][1]) + (new_img.shape[0]-img2.shape[0])/2)), (R, G, B), 1)
    cv2.imshow("Matches", new_img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    # 打印实际匹配的特征点数
    print("SIFT Number of matches: ", good_sum)
    return new_img

# def register_images_surf(img1_path, img2_path):
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Create SURF detector
    surf = cv2.xfeatures2d.SURF_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = surf.detectAndCompute(gray1, None)
    kp2, des2 = surf.detectAndCompute(gray2, None)

    # Create FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find matches
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_matches

def register_images_orb(img1_path, img2_path, matrix):
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Create ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Create BFMatcher (Brute Force Matcher)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    print("ORB Origin matches: ", len(matches))

    # cv2.imshow("Origin_Matches", img_matches)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Apply transformation matrix
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    # 添加一列全为 1 的值，以表示齐次坐标
    src_pts_homogeneous = np.hstack((src_pts.squeeze(), np.ones((len(src_pts), 1))))
    # 扩展矩阵维度，以便进行矩阵乘法
    matrix = np.vstack((matrix, [0, 0, 1]))
    # 计算变换后的点坐标
    transformed_src_pts = np.dot(matrix, src_pts_homogeneous.T)
    # 将齐次坐标转换为二维坐标
    transformed_src_pts = transformed_src_pts[:2, :].T

    good_sum = 0
    new_src = []
    new_dst = []
    for i in range(len(transformed_src_pts)):
        dist = np.linalg.norm(transformed_src_pts[i] - dst_pts[i])
        # 如果距离小于 10，认为是正确的匹配
        if dist < 10:
            if transformed_src_pts[i][0] < 0 or transformed_src_pts[i][1] < 0:
                continue
            if transformed_src_pts[i][0] > img1.shape[1] or transformed_src_pts[i][1] > img1.shape[0]:
                continue
            good_sum += 1
            new_src.append(src_pts[i])
            new_dst.append(dst_pts[i])
    src_pts = np.array(new_src)
    dst_pts = np.array(new_dst)
    #绘制实际上的匹配点
    # 创建一个新的图像，将两幅输入图像拼接到一起
    new_img = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), np.uint8)
    new_img[:img1.shape[0], :img1.shape[1]] = img1
    #
    new_img[int((new_img.shape[0]-img2.shape[0])/2):int((new_img.shape[0]+img2.shape[0])/2), img1.shape[1]:] = img2
    # 使用cv2.circle()在图像上绘制点, cv2.line()绘制线
    for i in range(len(src_pts)):
        R = random.randint(0, 255)
        G = random.randint(0, 255)
        B = random.randint(0, 255)
        cv2.circle(new_img, (int(src_pts[i][0][0]), int(src_pts[i][0][1])), 2, (R, G, B), 2)
        cv2.circle(new_img, (int(dst_pts[i][0][0]) + img1.shape[1], int((dst_pts[i][0][1]) + (new_img.shape[0]-img2.shape[0])/2)), 2, (R, G, B), 2)
        cv2.line(new_img, (int(src_pts[i][0][0]), int(src_pts[i][0][1])), (int(dst_pts[i][0][0]) + img1.shape[1], int((dst_pts[i][0][1]) + (new_img.shape[0]-img2.shape[0])/2)), (R, G, B), 1)
        # cv2.circle(new_img, (int(src_pts[i][0][0]), int(src_pts[i][0][1])), 2, (R, G, B), 2)
        # cv2.circle(new_img, (int(transformed_src_pts[i][0]) + img1.shape[1], int((transformed_src_pts[i][1]) + (new_img.shape[0]-img2.shape[0])/2), 2, (R, G, B), 2)
        # cv2.line(new_img, (int(src_pts[i][0][0]), int(src_pts[i][0][1]), (int(transformed_src_pts[i][0]) + img1.shape[1], int((transformed_src_pts[i][1]) + (new_img.shape[0]-img2.shape[0])/2), (R, G, B), 1)
    
    cv2.imshow("Matches", new_img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    # Draw matches
    # img_matches = cv2.drawMatches(img1, kp1, img2, kp2, new_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # 打印实际匹配的特征点数
    print("ORB Number of matches: ", good_sum)
    return new_img

def read_matrix(filename):
    # 读取文件内容
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # 解析变换矩阵
    transformation_matrix = []
    for line in lines:
        # 按空格分割并将字符串转换为浮点数
        row = list(map(float, line.strip().split()))
        transformation_matrix.append(row)
    
    # 将列表转换为 numpy 数组
    transformation_matrix = np.array(transformation_matrix)

    return transformation_matrix


if __name__ == "__main__":
    img1_path = './images/Veluwe/Veluwe.png'
    img2_path = './images/Veluwe/Veluwe_IR.png'
    os.makedirs('./images/output/Veluwe', exist_ok=True)
    matrix = read_matrix('./images/Veluwe/Veluwe.txt')
    SIFT_registered_image = register_images_sift(img1_path, img2_path, matrix)
    # SURF_registered_image = register_images_surf(img1_path, img2_path)
    ORB_registered_image = register_images_orb(img1_path, img2_path, matrix)
    cv2.imwrite('./images/output/Veluwe/SIFT_Registered.png', SIFT_registered_image)
    cv2.imwrite('./images/output/Veluwe/ORB_Registered.png', ORB_registered_image)

