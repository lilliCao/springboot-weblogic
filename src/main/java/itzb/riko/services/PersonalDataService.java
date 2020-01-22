package itzb.riko.services;

import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.highgui.Highgui;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;

@Service
@Slf4j
public class PersonalDataService {

    private IFaceDetector faceDetector;

    private ILicensePlateDetector licensePlateDetector;

    @PostConstruct
    private void init() {
        nu.pattern.OpenCV.loadShared();
        this.faceDetector = new HaarCascadeFaceDetector();
        this.licensePlateDetector = new EdgeLicensePlateDetector();
    }

    public void setFaceDetector(IFaceDetector faceDetector) {
        this.faceDetector = faceDetector;
    }

    public void setLicensePlateDetector(ILicensePlateDetector licensePlateDetector) {
        this.licensePlateDetector = licensePlateDetector;
    }

    public byte[] clearPersonalData(byte[] bytes) {
        Mat mat = byteToMat(bytes);
        Mat returnMat = mat;
        Rect[] faces = this.faceDetector.findFace(mat);
        MatOfPoint licensePlate = this.licensePlateDetector.findLicensePlate(mat);
        if (faces.length > 0) {
            returnMat = this.faceDetector.blackFace(mat, faces);
        }
        if (licensePlate != null) {
            returnMat = this.licensePlateDetector.blackLicensePlate(mat, licensePlate);
        }
        return matToByte(returnMat);
    }

    public byte[] clearFace(byte[] bytes) {
        Mat mat = byteToMat(bytes);
        Mat returnMat = mat;
        Rect[] faces = this.faceDetector.findFace(mat);
        if (faces.length > 0) {
            returnMat = this.faceDetector.blackFace(mat, faces);
        }
        return matToByte(returnMat);
    }

    public byte[] clearLicensePlate(byte[] bytes) {
        Mat mat = byteToMat(bytes);
        Mat returnMat = mat;
        MatOfPoint licensePlate = this.licensePlateDetector.findLicensePlate(mat);
        if (licensePlate != null) {
            returnMat = this.licensePlateDetector.blackLicensePlate(mat, licensePlate);
        }
        return matToByte(returnMat);
    }

    private Mat byteToMat(byte[] bytes) {
        return Highgui.imdecode(new MatOfByte(bytes), Highgui.IMREAD_UNCHANGED);
    }

    private byte[] matToByte(Mat mat) {
        MatOfByte matOfByte = new MatOfByte();
        Highgui.imencode(".png", mat, matOfByte);
        return matOfByte.toArray();
    }
}
