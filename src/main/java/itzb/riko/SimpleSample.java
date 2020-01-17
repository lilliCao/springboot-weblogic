package itzb.riko;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SimpleSample {

    public static void main(String[] args) {
        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        nu.pattern.OpenCV.loadShared();
        String path = "C:\\Users\\thicao\\Desktop\\l3i2\\idea_projects\\demo\\personaldata\\src\\main\\resources\\test\\";
        testLicense(path, "demo.jpg", "output.jpg");
        /*testLicense(path, "demo1.jpg", "output1.jpg");
        testLicense(path, "demo2.jpg", "output2.jpg");
        testLicense(path, "test_007.jpg", "output_007.jpg");
        testLicense(path, "test_013.jpg", "output_013.jpg");
        testLicense(path, "test_020.jpg", "output_020.jpg");
        testLicense(path, "test_021.jpg", "output_021.jpg");

         */
    }

    private static void test() {
    }

    public static void testLicense(String path, String input, String output) {
        float scoreThresh = 0.5f;
        float nmsThresh = 0.4f;
        Net net = Dnn.readNetFromTensorflow("C:\\Users\\thicao\\Desktop\\l3i2\\idea_projects\\demo\\personaldata\\src\\main\\resources\\frozen_east_text_detection.pb");

        // input image

        Mat img = Imgcodecs.imread(path + input);
        Mat frame = new Mat();

        Imgproc.cvtColor(img, frame, Imgproc.COLOR_RGBA2RGB);
        Mat blob = Dnn.blobFromImage(frame, 1.0, new Size(320, 320), new Scalar(123.68, 116.78, 103.94), true, false);
        net.setInput(blob);

        List<Mat> outs = new ArrayList();
        List<String> outNames = Arrays.asList("feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3");
        net.forward(outs, outNames);

        // scores.shape : (1,1,80,80)
        Mat scores = outs.get(0).reshape(1, 80);
        // geometry.shape : (1,5,80,80)
        // My lord and savior : http://answers.opencv.org/question/175676/javaandroid-access-4-dim-mat-planes/
        Mat geometry = outs.get(1).reshape(1, 5 * 80);
        List<Float> confidencesList = new ArrayList<>();
        // Decode predicted bounding boxes.
        List<RotatedRect> boxesList = decode(scores, geometry, confidencesList);

        float[] confidencesArray = new float[confidencesList.size()];
        for (int i = 0; i < confidencesList.size(); ++i) {
            confidencesArray[i] = confidencesList.get(i) != null ? confidencesList.get(i) : Float.NaN;
        }

        for (RotatedRect rotatedRect : boxesList) {
            drawRotatedRect(img, rotatedRect, new Scalar(0, 255, 0), 4);
        }
        Imgcodecs.imwrite(path + output, img);
    }

    public static void drawRotatedRect(Mat image, RotatedRect rotatedRect, Scalar color, int thickness) {
        Point[] vertices = new Point[4];
        rotatedRect.points(vertices);
        MatOfPoint points = new MatOfPoint(vertices);
        Imgproc.drawContours(image, Arrays.asList(points), -1, color, thickness);
    }

    private static List<RotatedRect> decode(Mat scores, Mat geometry, List<Float> confidences) {
        if (scores.dims() != 2 && geometry.dims() != 2 && scores.height() != 80 &&
                scores.width() != 80 && geometry.height() != 400 && geometry.width() != 80) {
            throw new RuntimeException("That sucks mate");
        }
        List<RotatedRect> detections = new ArrayList<>();
        float scoreThresh = 0.5f;
        for (int y = 0; y < 80; ++y) {
            Mat scoresData = scores.row(y);
            //1st plane
            Mat x0Data = geometry.submat(0, 80, 0, 80).row(y);
            //2nd plane
            Mat x1Data = geometry.submat(80, 2 * 80, 0, 80).row(y);
            Mat x2Data = geometry.submat(2 * 80, 3 * 80, 0, 80).row(y);
            Mat x3Data = geometry.submat(3 * 80, 4 * 80, 0, 80).row(y);
            Mat anglesData = geometry.submat(4 * 80, 5 * 80, 0, 80).row(y);

            for (int x = 0; x < 80; ++x) {
                double score = scoresData.get(0, x)[0];
                if (score >= scoreThresh) {
                    double offsetX = x * 4.0;
                    double offsetY = y * 4.0;
                    double angle = anglesData.get(0, x)[0];
                    double cosA = Math.cos(angle);
                    double sinA = Math.sin(angle);
                    double x0 = x0Data.get(0, x)[0];
                    double x1 = x1Data.get(0, x)[0];
                    double x2 = x2Data.get(0, x)[0];
                    double x3 = x3Data.get(0, x)[0];
                    double h = x0 + x2;
                    double w = x1 + x3;
                    Point offset = new Point(offsetX + cosA * x1 + sinA * x2, offsetY - sinA * x1 + cosA * x2);
                    Point p1 = new Point(-1 * sinA * h + offset.x, -1 * cosA * h + offset.y);
                    Point p3 = new Point(-1 * sinA * w + offset.x, -1 * cosA * w + offset.y);
                    RotatedRect r = new RotatedRect(new Point(0.5 * (p1.x + p3.x), 0.5 * (p1.y + p3.y)), new Size(w, h), -1 * angle * 180 / Math.PI);
                    detections.add(r);
                    confidences.add((float) score);
                }
            }
        }
        return detections;
    }
}
