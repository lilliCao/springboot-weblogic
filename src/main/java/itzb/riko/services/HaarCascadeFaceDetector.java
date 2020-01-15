package itzb.riko.services;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;

@Slf4j
public class HaarCascadeFaceDetector implements IFaceDetector {
    private final String HAAR_CASCADE_FRONTAL_FACE = "haarcascade_frontalface_default.xml";
    private final Scalar BLACK = new Scalar(0, 0, 0);
    private final int THICKNESS = -1;
    private CascadeClassifier faceDetector;

    public HaarCascadeFaceDetector() {
        this.faceDetector = new CascadeClassifier();
        boolean load;
        try {
            File file = File.createTempFile("haar_cascade_pre_trained", ".tmp");
            OutputStream outputStream = new FileOutputStream(file);
            IOUtils.copy(Thread.currentThread().getContextClassLoader().getResourceAsStream(HAAR_CASCADE_FRONTAL_FACE), outputStream);
            String path = file.getAbsolutePath();
            load = this.faceDetector.load(path);
            outputStream.close();
            file.deleteOnExit();
        } catch (Exception e) {
            log.error("Can not load haarcascade face detector file");
            return;
        }
        if (!load) {
            log.error("Can not load haarcascade face detector file properly");
            return;
        }
    }

    @Override
    public Rect[] findFace(Mat mat) {
        MatOfRect faceDetections = new MatOfRect();
        this.faceDetector.detectMultiScale(mat, faceDetections);
        return faceDetections.toArray();
    }

    @Override
    public Mat blackFace(Mat mat, Rect[] faces) {
        for (Rect rect : faces) {
            Imgproc.rectangle(mat, new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height),
                    this.BLACK,
                    this.THICKNESS);
        }
        return mat;
    }
}
