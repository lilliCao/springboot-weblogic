package itzb.riko.services;

import org.opencv.core.Mat;
import org.opencv.core.Rect;

public interface IFaceDetector {
    public Rect[] findFace(Mat mat);

    public Mat blackFace(Mat mat, Rect[] faces);
}
