package itzb.riko;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

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
}
