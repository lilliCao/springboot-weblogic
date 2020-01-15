package itzb.riko.services;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class EdgeLicensePlateDetector implements ILicensePlateDetector {
    private final Scalar BLACK = new Scalar(0, 0, 0);
    private final int THICKNESS = -1;

    @Override
    public MatOfPoint findLicensePlate(Mat mat) {
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

    @Override
    public Mat blackLicensePlate(Mat mat, MatOfPoint rect) {
        Imgproc.drawContours(mat, Arrays.asList(rect), -1, BLACK, THICKNESS);
        return mat;
    }

    private boolean isAcceptableRect(Rect rect, Mat mat) {
        double ratio = rect.size().width / rect.size().height;
        return ratio > 3 && ratio < 5 && rect.area() > mat.size().area() / 100;
    }
}
