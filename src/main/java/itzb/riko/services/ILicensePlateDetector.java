package itzb.riko.services;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;

public interface ILicensePlateDetector {
    public MatOfPoint findLicensePlate(Mat mat);

    public Mat blackLicensePlate(Mat mat, MatOfPoint rect);
}
