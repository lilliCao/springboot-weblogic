package itzb.riko.services;

import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Point;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.beans.factory.annotation.Autowired;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Slf4j
public class EdgeLicensePlateDetector implements ILicensePlateDetector {

    private final Scalar BLACK = new Scalar(0, 0, 0);
    private final int THICKNESS = -1;
    private OCRService ocrService;

    @Autowired
    public EdgeLicensePlateDetector(OCRService ocrService) {
        this.ocrService = ocrService;
    }


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
        BufferedImage bufferedImage = matToBufferedImage(mat);
        String licenseText = findTextInRectangle(bufferedImage, license.boundingRect());
        for (int i = 1; i < contours.size(); i++) {
            MatOfPoint contour = contours.get(i);
            RotatedRect rotatedRect = Imgproc.minAreaRect(new MatOfPoint2f(contour.toArray()));
            Rect rect = rotatedRect.boundingRect();
            if (isAcceptableRect(rect, mat)) {
                String newText = findTextInRectangle(bufferedImage, rect);
                if (newText == null) {
                    if (rect.size().area() > license.boundingRect().size().area()) {
                        license = rotatedRect;
                    }
                } else {
                    if (newText.length() > licenseText.length()) {
                        license = rotatedRect;
                        licenseText = newText;
                    }
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
        boolean isNotTooBig = mat.size().area() / 7 > rect.size().area();
        boolean isNotTooSmall = rect.area() > mat.size().area() / 100;
        double ratio = rect.size().width / rect.size().height;
        return ratio > 3 && ratio < 5 && isNotTooBig && isNotTooSmall;
    }

    private BufferedImage matToBufferedImage(Mat mat) {
        MatOfByte matOfByte = new MatOfByte();
        Imgcodecs.imencode(".png", mat, matOfByte);
        byte[] byteArray = matOfByte.toArray();
        BufferedImage bufImage = null;
        try {
            InputStream in = new ByteArrayInputStream(byteArray);
            bufImage = ImageIO.read(in);
        } catch (Exception e) {
            log.info("Failed to convert mat to buffered image for ocr task");
        }
        return bufImage;
    }

    private String findTextInRectangle(BufferedImage image, Rect rect) {
        Rectangle rectangle = new Rectangle(rect.x, rect.y, rect.width, rect.height);
        try {
            String text = ocrService.findText(image, rectangle);
            return text.length() < 5 ? null : text.replaceAll("[^0-9a-zA-Z]", "");
        } catch (Exception e) {
            log.info("Failed to detect text in given rectangle");
            return null;
        }
    }
}
