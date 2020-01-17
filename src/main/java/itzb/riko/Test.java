package itzb.riko;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Test {
    public static void main(String[] args) {
        nu.pattern.OpenCV.loadShared();
        String path = "C:\\Users\\thicao\\Desktop\\l3i2\\idea_projects\\demo\\personaldata\\src\\main\\resources\\test\\";
        String pathResource = "C:\\Users\\thicao\\Desktop\\l3i2\\idea_projects\\demo\\personaldata\\src\\main\\resources\\";
        testCar(pathResource, "demo.jpg", "output.jpg");
        testCar(pathResource, "demo1.jpg", "output1.jpg");
        testCar(pathResource, "demo2.jpg", "output2.jpg");
        testCar(pathResource, "test_007.jpg", "output_007.jpg");
        testCar(pathResource, "test_013.jpg", "output_013.jpg");
        testCar(pathResource, "test_020.jpg", "output_020.jpg");
        testCar(pathResource, "test_021.jpg", "output_021.jpg");
        /*
        testLicense(path, "demo.jpg", "output.jpg");
        testLicense(path, "demo1.jpg", "output1.jpg");
        testLicense(path, "demo2.jpg", "output2.jpg");
        testLicense(path, "test_007.jpg", "output_007.jpg");
        testLicense(path, "test_013.jpg", "output_013.jpg");
        testLicense(path, "test_020.jpg", "output_020.jpg");
        testLicense(path, "test_021.jpg", "output_021.jpg");

         */

    }

    private static void testText(String path, String input, String output) {
        // TODO https://stackoverflow.com/questions/53402064/opencv-east-text-detector-implementation-in-java
    }

    private static void testCar(String pathResource, String input, String output) {

        CascadeClassifier car_cascade = new CascadeClassifier();
        boolean load = car_cascade.load(pathResource + "haarcascade_licence_plate_rus_16stages.xml");
        if (!load) {
            System.out.println("Error loading");
            return;
        }
        String path = pathResource + "test\\";

        Mat mat = Imgcodecs.imread(path + input);

        MatOfRect faceDetections = new MatOfRect();
        car_cascade.detectMultiScale(mat, faceDetections);
        if (faceDetections.toArray().length < 1) {
            System.out.println("Not found");
            return;
        }
        for (Rect rect : faceDetections.toArray()) {
            Imgproc.rectangle(mat, new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 0, 0),
                    -1);
        }
        Imgcodecs.imwrite(path + output, mat);

    }

    private static void testLicense(String path, String input, String output) {
        Mat mat = Imgcodecs.imread(path + input);
        //Mat mat = new Mat();
        //Imgproc.resize(matOri, mat, new Size(500, 500));

        // gray image
        Mat imgGray = new Mat();
        Imgproc.cvtColor(mat, imgGray, Imgproc.COLOR_BGR2GRAY);
        //Imgcodecs.imwrite(path+"gray.jpg", imgGray);

        // guassian blur
        Mat imgGuassianBlur = new Mat();
        Imgproc.GaussianBlur(imgGray, imgGuassianBlur, new Size(5, 5), 0);
        //Imgcodecs.imwrite(path+"guassianBlur.jpg", imgGuassianBlur);

        // sobel for edge detection (Canny)
        Mat imgSobel = new Mat();
        Imgproc.Sobel(imgGuassianBlur, imgSobel, -1, 1, 0);
        //Imgcodecs.imwrite(path+"sobel.jpg", imgSobel);

        // threshold
        Mat imgThreshold = new Mat();
        Imgproc.threshold(imgSobel, imgThreshold, 200, 255, Imgproc.THRESH_OTSU);
        //Imgcodecs.imwrite(path+"threshold.jpg", imgThreshold);

        // close by erosion to reduce noise
        Mat kernel = Imgproc.getStructuringElement(Imgproc.CV_SHAPE_RECT, new Size(22, 8));
        Mat imgClose = new Mat();
        Imgproc.morphologyEx(imgThreshold, imgClose, 1, kernel);
        //Imgcodecs.imwrite(path+"ouput.jpg", imgClose);

        // find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(imgClose, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        Scalar white = new Scalar(255, 0, 0);
        Scalar color1 = new Scalar(255, 192, 203);
        Scalar color2 = new Scalar(85, 50, 40);
        RotatedRect license = Imgproc.minAreaRect(new MatOfPoint2f(contours.get(0).toArray()));
        for (int i = 1; i < contours.size(); i++) {
            MatOfPoint contour = contours.get(i);
            // color all contour
            Imgproc.fillPoly(mat, Arrays.asList(contour), white);
            RotatedRect rotatedRect = Imgproc.minAreaRect(new MatOfPoint2f(contour.toArray()));
            Rect rect = rotatedRect.boundingRect();
            drawRotatedRect(mat, rotatedRect, new Scalar(0, 0, 0), 1);
            float ratio = getRatio(rect);
            if (ratio > 3 && ratio < 5 && rect.area() > mat.size().area() / 100) {
                // color all possible license
                drawRotatedRect(mat, rotatedRect, color1, 4);
                if (rect.size().area() > license.boundingRect().size().area() && checkSize(rect, mat)) {
                    license = rotatedRect;
                }
            }
        }
        // draw the real license
        drawRotatedRect(mat, license, color2, -1);
        Imgcodecs.imwrite(path + output, mat);
    }

    private static float getRatio(Rect rect) {
        return (float) rect.size().width / (float) rect.size().height;
    }

    private static boolean checkSize(Rect rect, Mat mat) {
        return mat.size().area() / 7 > rect.size().area();
    }

    public static void drawRotatedRect(Mat image, RotatedRect rotatedRect, Scalar color, int thickness) {
        Point[] vertices = new Point[4];
        rotatedRect.points(vertices);
        MatOfPoint points = new MatOfPoint(vertices);
        Imgproc.drawContours(image, Arrays.asList(points), -1, color, thickness);
    }
}
