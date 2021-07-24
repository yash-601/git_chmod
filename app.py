import base64

from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "hello world"


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


@app.route('/corner_detection', methods=['POST'])
def corner_detection():
    data = request.get_json(force=True)
    image_data = data['image']
    imgdata = base64.b64decode(image_data)
    pilImage = Image.open(io.BytesIO(imgdata))

    img = cv2.cvtColor(np.array(pilImage), cv2.COLOR_RGB2BGR)
    # pilImage.show()
    # cv2.imshow("soham",img)
    # cv2.waitKey(10000)
    # img = cv2.imread('C:\\Users\\soham\\OneDrive\\Desktop\\in.jpeg')

    heightImg = img.shape[0]
    widthImg = img.shape[1]

    # img = cv2.resize(img, (widthImg, heightImg))
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    thres = [200, 200]  # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])  # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION

    ## FIND ALL COUNTOURS
    imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # DRAW ALL DETECTED CONTOURS

    # FIND THE BIGGEST COUNTOUR
    biggest, maxArea = biggestContour(contours)  # FIND THE BIGGEST CONTOUR
    if biggest.size != 0:
        biggest = reorder(biggest)
        output = []
        for i in biggest:
            output.append(int(i[0][0]))
            output.append(int(i[0][1]))
            print(i[0][0], i[0][1])

        output.append(heightImg)
        output.append(widthImg)
        return jsonify(output)
        # cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
        # imgBigContour = drawRectangle(imgBigContour, biggest, 2)
        # pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
        # pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
        # matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        #
        # # REMOVE 20 PIXELS FORM EACH SIDE
        # imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        # imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))
        #
        # # APPLY ADAPTIVE THRESHOLD
        # imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        # imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        # imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        # imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)
    else:
        return jsonify([0, 0, widthImg, 0, 0, heightImg, widthImg, heightImg])


@app.route('/cropper', methods=['POST'])
def cropper():
    data = request.get_json(force=True)
    points = data['points']
    image_data = data['image']
    imgdata = base64.b64decode(image_data)
    pilImage = Image.open(io.BytesIO(imgdata))
    img = cv2.cvtColor(np.array(pilImage), cv2.COLOR_RGB2BGR)

    heightImg = img.shape[0]
    widthImg = img.shape[1]

    print(points)
    pts1 = np.float32(points)  # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    # cv2.imshow("soham",imgWarpColored)
    # cv2.waitKey(10000)
    # return jsonify(image=base64.b64encode(b.read()).decode('ascii'))
    io_buf = io.BytesIO(imgWarpColored)
    image_bytes = base64.b64encode(io_buf.read()).decode('ascii')
    print(type(image_bytes), type(imgdata), len(imgdata), len(image_bytes))
    # imgdata = base64.b64decode(image_data)
    # pilImage = Image.open(io.BytesIO(imgdata))
    # img = cv2.cvtColor(np.array(pilImage), cv2.COLOR_RGB2BGR)
    # cv2.imshow("soham",img)
    # cv2.waitKey(10000)
    # print(image_bytes)
    data_object = {"image": base64.b64encode(io_buf.read()).decode('ascii')}
    return jsonify(image=image_bytes)


if __name__ == "__main__":
    app.run()
