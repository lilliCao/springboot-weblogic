package itzb.riko;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Service
@Slf4j
public class PersonalDataService {

    @Value("haarcascade_frontalface_default.xml")
    private String haarFrontalFaceFile;

    private CascadeClassifier faceDetector;

    @PostConstruct
    private void init() {
        nu.pattern.OpenCV.loadShared();
        this.faceDetector = new CascadeClassifier();
        boolean load;
        try {
            File file = new File("temp");
            OutputStream outputStream = new FileOutputStream(file);
            IOUtils.copy(Thread.currentThread().getContextClassLoader().getResourceAsStream(this.haarFrontalFaceFile), outputStream);
            String path = file.getAbsolutePath();
            load = this.faceDetector.load(path);
        } catch (Exception e) {
            log.error("Can not load haarcascade face detector file");
            return;
        }
        if (!load) {
            log.error("Can not load haarcascade face detector file properly");
            return;
        }
    }

    public byte[] clearFace(byte[] bytes) {
        Mat mat = byteToMat(bytes);
        Mat returnMat = mat;
        Rect[] faces = findFace(mat);
        if (faces.length > 0) {
            returnMat = blackFace(mat, faces);
        }
        return matToByte(returnMat);
    }

    public byte[] clearLicensePlate(byte[] bytes) {
        Mat mat = byteToMat(bytes);
        Mat returnMat = mat;
        MatOfPoint licensePlate = findLicensePlate(mat);
        if (licensePlate != null) {
            returnMat = blackLicensePlate(mat, licensePlate);
        }
        return matToByte(returnMat);
    }

    private Mat byteToMat(byte[] bytes) {
        return Imgcodecs.imdecode(new MatOfByte(bytes), Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
    }

    private byte[] matToByte(Mat mat) {
        MatOfByte matOfByte = new MatOfByte();
        Imgcodecs.imencode(".png", mat, matOfByte);
        return matOfByte.toArray();
    }

    private Rect[] findFace(Mat mat) {
        MatOfRect faceDetections = new MatOfRect();
        this.faceDetector.detectMultiScale(mat, faceDetections);
        return faceDetections.toArray();
    }

    private Mat blackFace(Mat mat, Rect[] faces) {
        for (Rect rect : faces) {
            Imgproc.rectangle(mat, new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 0, 0),
                    -1);
        }
        return mat;
    }

    private Mat blackLicensePlate(Mat mat, MatOfPoint rect) {
        Imgproc.drawContours(mat, Arrays.asList(rect), -1, new Scalar(0, 0, 0), -1);
        return mat;
    }

    private MatOfPoint findLicensePlate(Mat mat) {
        Mat imgGray = new Mat();
        Imgproc.cvtColor(mat, imgGray, Imgproc.COLOR_BGR2GRAY);

        Mat imgGuassianBlur = new Mat();
        Imgproc.GaussianBlur(imgGray, imgGuassianBlur, new Size(5, 5), 0);

        Mat imgSobel = new Mat();
        Imgproc.Sobel(imgGuassianBlur, imgSobel, -1, 1, 0);

        Mat imgThreshold = new Mat();
        Imgproc.threshold(imgSobel, imgThreshold, 200, 255, Imgproc.THRESH_OTSU);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.CV_SHAPE_RECT, new Size(22, 8));
        Mat imgClose = new Mat();
        Imgproc.morphologyEx(imgThreshold, imgClose, 1, kernel);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(imgClose, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        RotatedRect license = Imgproc.minAreaRect(new MatOfPoint2f(contours.get(0).toArray()));
        for (int i = 1; i < contours.size(); i++) {
            MatOfPoint contour = contours.get(i);
            RotatedRect rotatedRect = Imgproc.minAreaRect(new MatOfPoint2f(contour.toArray()));
            Rect rect = rotatedRect.boundingRect();
            boolean isNotTooBig = mat.size().area() / 7 > rect.size().area();
            if (isAcceptableRect(rect, mat)) {
                if (rect.size().area() > license.boundingRect().size().area() && isNotTooBig) {
                    license = rotatedRect;
                }
            }
        }
        if (isAcceptableRect(license.boundingRect(), mat)) {
            Point[] vertices = new Point[4];
            license.points(vertices);
            MatOfPoint returnLicense = new MatOfPoint(vertices);
            return returnLicense;
        } else {
            return null;
        }
    }

    private boolean isAcceptableRect(Rect rect, Mat mat) {
        double ratio = rect.size().width / rect.size().height;
        return ratio > 3 && ratio < 5 && rect.area() > mat.size().area() / 100;
    }
}
