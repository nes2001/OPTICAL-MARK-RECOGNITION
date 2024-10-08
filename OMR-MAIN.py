import cv2
import numpy as np
import sys
import os

# Add module directory path
sys.path.append(os.path.abspath("your-module-directory-path(utlis.py)"))

import utlis

# your image path
image_path = 'your-image-path'

if not os.path.exists(image_path):
    print(f"Dosya bulunamadı: {image_path}")
else:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Dosya okunamadı: {image_path}")
    else:
        widthImg, heightImg = 600, 700  # Örnek boyutlar
        questions = 10
        choices = 5
        
        ans = [1, 1, 0, 0, 1, 0, 1, 0, 0, 0]
        
        img = cv2.resize(img, (widthImg, heightImg))
        imgContours = img.copy()
        imgBiggestContours = img.copy()

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        imgCanny = cv2.Canny(imgBlur, 10, 50)

        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 5)

        rectCon = utlis.rectContour(contours)
        biggestContour = utlis.getCornerPoints(rectCon[0])
        
        if biggestContour.size != 0:
            cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 20)
            biggestContour = utlis.reorder(biggestContour)
            
            pt1 = np.float32(biggestContour)
            pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pt1, pt2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            # Sağ ve sol kenar konturlarını sayma
            def count_contours_side(image, side):
                side_img = image[:, :image.shape[1] // 2] if side == 'left' else image[:, image.shape[1] // 2:]
                contours, _ = cv2.findContours(side_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                return len(contours)

            left_contours = count_contours_side(imgCanny, 'left')
            right_contours = count_contours_side(imgCanny, 'right')

            print(f"Left Contours: {left_contours}, Right Contours: {right_contours}")

            if right_contours > left_contours:
                imgWarpColored = cv2.rotate(imgWarpColored, cv2.ROTATE_180)
                print("Image rotated by 180 degrees.")

            # Threshold uygula
            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpGray, 110, 255, cv2.THRESH_BINARY_INV)[1]

            boxes = utlis.splitBoxes(imgThresh)

            # Her kutudaki piksel sayılarını al
            myPixelVal = np.zeros((questions, choices), dtype=np.uint16)
            countC = 0
            countR = 0

            for image in boxes:
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC] = totalPixels
                countC += 1
                if countC == choices:
                    countR += 1
                    countC = 0

            print("Pixel Values:\n", myPixelVal)

            # Find index values
            myIndex = []

            for x in range(questions):
                arr = myPixelVal[x]

                # Mark as a blank question if all options are less than a certain threshold value
                if np.all(arr < 2000):
                    myIndex.append(-1)
                else:
                    # Count the choices greater than the threshold value
                    high_pixels = np.where(arr >= 3000)[0]

                    if len(high_pixels) > 1:
                        myIndex.append(-2)  # Invalid question if more than one choice is greater than a certain threshold
                    else:
                        # If a single chic is greater than a certain threshold
                        max_index = np.argmax(arr)
                        if arr[max_index] >= 4000:
                            myIndex.append(max_index)  # Add the index of the correct option if it is greater than a certain high value
                        else:
                            myIndex.append(-2)  #Mark as invalid question if less than threshold

            print("İndex Values:\n", myIndex)

            # Grading
            grading = [1 if ans[x] == myIndex[x] else 0 for x in range(questions)]
            score = (sum(grading) / questions) * 100
            print("Grading:\n", grading)
            print(f"Score: {score}")

            # Show Answers
            imgResult = imgWarpColored.copy()
            imgResult = utlis.showAnswers(imgResult, myIndex, grading, ans, questions, choices)

            # Merge images
            imgBlank = np.zeros_like(img)
            imageArray = ([img, imgGray, imgBlur, imgCanny],
                          [imgContours, imgBiggestContours, imgWarpColored, imgResult])
            imgStacked = utlis.stackImages(imageArray, 0.5)

            cv2.imshow("Stacked Image", imgStacked)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
